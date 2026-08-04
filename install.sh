#!/usr/bin/env bash
#
# fsfs installer (frankensearch standalone CLI)
#
# One-liner install:
#   curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/frankensearch/main/install.sh | bash
#
# With cache buster:
#   curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/frankensearch/main/install.sh?$(date +%s)" | bash
#
# Options:
#   --version vX.Y.Z   Install specific version (default: latest)
#   --dest DIR         Install to DIR (default: ~/.local/bin)
#   --system           Install to /usr/local/bin (requires sudo)
#   --easy-mode        Auto-update PATH in shell rc files
#   --verify           Run self-test after install
#   --force            Reinstall even when the resolved version is already present
#   --offline          Never touch the network; requires --version and a local
#                      --artifact-url plus --checksum
#   --from-source      Build from source instead of downloading binary
#   --lite             Force the model-free source profile (~15MB binary)
#   --quiet            Suppress non-error output
#   --no-gum           Disable gum formatting even if available
#
# Build profiles:
#   Full release artifacts embed ML models for zero-config semantic search.
#   Cargo and source-build defaults compile Model2Vec + FastEmbed loaders but
#   acquire the pinned model bytes separately with `fsfs download-models`.
#   Use --lite to force the explicit model-free --no-default-features lane;
#   downloaded files alone cannot add loaders to that stripped binary.
#   Equivalent to: cargo build --release -p frankensearch-fsfs --no-default-features
#
set -euo pipefail
umask 022
shopt -s lastpipe 2>/dev/null || true

OWNER="${OWNER:-Dicklesworthstone}"
REPO="${REPO:-frankensearch}"
BINARY_NAME="fsfs"
VERSION="${VERSION:-}"
DEST_DEFAULT="$HOME/.local/bin"
DEST="${DEST:-$DEST_DEFAULT}"
EASY=0
QUIET=0
VERIFY=0
FROM_SOURCE=0
LITE=0
FORCE=0
OFFLINE=0
CHECKSUM="${CHECKSUM:-}"
CHECKSUM_URL="${CHECKSUM_URL:-}"
ARTIFACT_URL="${ARTIFACT_URL:-}"
LOCK_FILE="${FSFS_INSTALL_LOCK_FILE:-/tmp/fsfs-install.lock}"
SYSTEM=0
NO_GUM=0
NO_COLOR_MODE=0
if [ -n "${NO_COLOR:-}" ]; then
  NO_COLOR_MODE=1
fi

# Preflight free-space floors (MB). The destination only holds the binary; the
# staging area holds the downloaded archive plus its extraction, or the whole
# source checkout and its Cargo target directory.
MIN_DEST_MB="${FSFS_INSTALL_MIN_DEST_MB:-128}"
MIN_STAGE_ARTIFACT_MB="${FSFS_INSTALL_MIN_STAGE_ARTIFACT_MB:-512}"
MIN_STAGE_SOURCE_MB="${FSFS_INSTALL_MIN_STAGE_SOURCE_MB:-2048}"

# Detect gum for fancy output (https://github.com/charmbracelet/gum)
HAS_GUM=0
if command -v gum &> /dev/null && [ -t 1 ]; then
  HAS_GUM=1
fi

log() { [ "$QUIET" -eq 1 ] && return 0; echo -e "$@"; }

info() {
  [ "$QUIET" -eq 1 ] && return 0
  if [ "$NO_COLOR_MODE" -eq 1 ]; then
    printf '%s\n' "→ $*"
  elif [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 39 "→ $*"
  else
    echo -e "\033[0;34m→\033[0m $*"
  fi
}

ok() {
  [ "$QUIET" -eq 1 ] && return 0
  if [ "$NO_COLOR_MODE" -eq 1 ]; then
    printf '%s\n' "✓ $*"
  elif [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 42 "✓ $*"
  else
    echo -e "\033[0;32m✓\033[0m $*"
  fi
}

warn() {
  [ "$QUIET" -eq 1 ] && return 0
  if [ "$NO_COLOR_MODE" -eq 1 ]; then
    printf '%s\n' "⚠ $*"
  elif [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 214 "⚠ $*"
  else
    echo -e "\033[1;33m⚠\033[0m $*"
  fi
}

# Errors always go to stderr regardless of renderer: --quiet suppresses routine
# output but a caller piping stdout must never receive a failure message as data.
err() {
  if [ "$NO_COLOR_MODE" -eq 1 ]; then
    printf '%s\n' "✗ $*" >&2
  elif [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 196 "✗ $*" >&2
  else
    echo -e "\033[0;31m✗\033[0m $*" >&2
  fi
}

validate_sha256() {
  local checksum="${1:-}"
  [ "${#checksum}" -eq 64 ] || return 1
  case "$checksum" in
    *[![:xdigit:]]*) return 1 ;;
  esac
}

checksum_from_manifest() {
  local manifest="$1" artifact="$2"
  [ -f "$manifest" ] || return 1
  awk -v artifact="$artifact" '
    length($1) == 64 && $1 ~ /^[[:xdigit:]]+$/ {
      filename = $2
      sub(/^\*/, "", filename)
      if (filename == artifact) {
        print tolower($1)
        exit
      }
    }
  ' "$manifest"
}

checksum_from_sidecar() {
  local sidecar="$1"
  [ -f "$sidecar" ] || return 1
  awk '
    length($1) == 64 && $1 ~ /^[[:xdigit:]]+$/ {
      print tolower($1)
      exit
    }
  ' "$sidecar"
}

verify_archive_checksum() {
  local archive="$1" expected="$2" tool_mode="${3:-auto}" actual=""
  local actual_normalized="" expected_normalized=""
  if ! validate_sha256 "$expected"; then
    err "Release checksum must be exactly 64 hexadecimal characters"
    return 1
  fi

  case "$tool_mode" in
    auto)
      if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum -- "$archive" | awk '{print $1}')
      elif command -v shasum >/dev/null 2>&1; then
        actual=$(shasum -a 256 -- "$archive" | awk '{print $1}')
      else
        err "No SHA-256 verifier found (need sha256sum or shasum); refusing to install an unverified artifact"
        return 1
      fi
      ;;
    none)
      # Used only by the non-networked contract test entrypoint below.
      err "No SHA-256 verifier found (need sha256sum or shasum); refusing to install an unverified artifact"
      return 1
      ;;
    *)
      err "Unknown checksum tool mode: $tool_mode"
      return 1
      ;;
  esac

  actual_normalized=$(printf '%s' "$actual" | tr '[:upper:]' '[:lower:]')
  expected_normalized=$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')
  if ! validate_sha256 "$actual" || [ "$actual_normalized" != "$expected_normalized" ]; then
    err "Checksum mismatch for $(basename "$archive")"
    return 1
  fi
}

install_route() {
  local explicit_lite="$1" full_artifact_available="$2" target="${3:-}"
  if [ "$explicit_lite" -eq 1 ]; then
    printf '%s\n' "source-lite"
  elif [ "$target" = "x86_64-apple-darwin" ]; then
    printf '%s\n' "unsupported-semantic"
  elif [ "$full_artifact_available" -eq 1 ]; then
    printf '%s\n' "artifact-full"
  else
    printf '%s\n' "source-default"
  fi
}

fail_unsupported_semantic_platform() {
  local target="$1"
  err "unsupported_platform: the ordinary semantic fsfs profile is not available for ${target}"
  err "The pinned ONNX Runtime distribution has no Intel macOS binary, so a default source fallback would fail."
  err "Use install.sh --lite for an explicit model-free build, or install the semantic profile on Apple Silicon or Linux."
  return 78
}

# --- Preflight ---------------------------------------------------------------
# Every check below runs before the destination is touched, so a rejected
# preflight always leaves the incumbent installation in place.

# Free megabytes on the filesystem that would hold $1, resolved through the
# nearest existing ancestor because the destination may not exist yet.
available_mb() {
  local path="$1" parent="" avail_kb=""
  [ -n "$path" ] || return 1
  while [ ! -e "$path" ]; do
    parent=$(dirname "$path")
    [ "$parent" = "$path" ] && break
    path="$parent"
  done
  [ -e "$path" ] || return 1
  # -P -k is the POSIX portable form; GNU and BSD df agree on field 4 = available.
  avail_kb=$(df -P -k "$path" 2>/dev/null | awk 'NR==2 {print $4}') || return 1
  case "$avail_kb" in
    ''|*[!0-9]*) return 1 ;;
  esac
  printf '%s\n' "$((avail_kb / 1024))"
}

check_disk_floor() {
  local path="$1" floor_mb="$2" label="$3" avail_mb=""
  if ! avail_mb=$(available_mb "$path"); then
    warn "Free space for $label ($path) is not measurable on this system; continuing without a floor check"
    return 0
  fi
  if [ "$avail_mb" -lt "$floor_mb" ]; then
    err "install.preflight.disk_space_low: $label ($path) has ${avail_mb}MB free but needs at least ${floor_mb}MB"
    err "Free space or select a larger volume, then rerun. The existing fsfs installation was not replaced."
    return 1
  fi
  info "Preflight: $label has ${avail_mb}MB free (floor ${floor_mb}MB)"
}

check_dest_writable() {
  local dest="$1"
  local probe="$dest"
  if [ "$SYSTEM" -eq 1 ]; then
    info "Preflight: --system install writes through sudo; skipping unprivileged write check"
    return 0
  fi
  local parent
  while [ ! -e "$probe" ]; do
    parent=$(dirname "$probe")
    [ "$parent" = "$probe" ] && break
    probe="$parent"
  done
  if [ ! -e "$probe" ]; then
    err "install.preflight.dest_unwritable: no existing ancestor of $dest could be resolved"
    return 1
  fi
  if [ ! -d "$probe" ]; then
    err "install.preflight.dest_unwritable: $probe exists but is not a directory"
    return 1
  fi
  if [ ! -w "$probe" ]; then
    err "install.preflight.dest_unwritable: $probe is not writable by the current user"
    err "Choose another --dest, fix the permissions, or rerun with --system to install through sudo."
    return 1
  fi
  info "Preflight: destination $dest is writable"
}

# Classifies whatever already occupies the destination path. Purely
# observational: it never writes and never removes the incumbent.
detect_existing_install() {
  local dest_binary="$1" resolved_version="$2" existing_version=""
  if [ ! -e "$dest_binary" ]; then
    printf '%s\n' "fresh"
    return 0
  fi
  if [ ! -x "$dest_binary" ]; then
    printf '%s\n' "unreadable"
    return 0
  fi
  if ! existing_version=$("$dest_binary" version 2>/dev/null | head -n 1); then
    printf '%s\n' "unreadable"
    return 0
  fi
  if [ -z "$existing_version" ]; then
    printf '%s\n' "unreadable"
    return 0
  fi
  case "$existing_version" in
    *"${resolved_version#v}"*) printf '%s\n' "same-version" ;;
    *) printf '%s\n' "different-version" ;;
  esac
}

# --offline forbids every network access, so the inputs that would otherwise be
# fetched must have been supplied explicitly.
check_offline_preconditions() {
  local offline="$1" version="$2" from_source="$3" artifact_url="$4" checksum="$5"
  [ "$offline" -eq 1 ] || return 0
  if [ -z "$version" ]; then
    err "install.preflight.offline_version_required: --offline needs an explicit --version vX.Y.Z because latest-release resolution queries GitHub"
    return 1
  fi
  if [ "$from_source" -eq 1 ]; then
    err "install.preflight.offline_source_unavailable: --offline cannot build from source; the source route clones the repository and fetches crates"
    return 1
  fi
  case "$artifact_url" in
    '')
      err "install.preflight.offline_artifact_required: --offline needs --artifact-url pointing at a locally available archive"
      return 1
      ;;
    http://*|https://*|ftp://*)
      err "install.preflight.offline_artifact_required: --artifact-url $artifact_url is a network URL; --offline needs a local path or file:// URL"
      return 1
      ;;
  esac
  if [ -z "$checksum" ]; then
    err "install.preflight.offline_checksum_required: --offline needs an explicit --checksum because the release checksum manifest cannot be fetched"
    return 1
  fi
  info "Preflight: offline inputs are complete"
}

# Reachability is reported, not enforced: an unreachable release endpoint is
# recoverable through the documented source-build fallback.
check_release_endpoint_reachable() {
  local url="$1"
  if [ "$OFFLINE" -eq 1 ]; then
    info "Preflight: --offline set; skipping release endpoint reachability probe"
    return 0
  fi
  if curl -fsS --connect-timeout 10 --max-time 20 -o /dev/null -I "$url" >/dev/null 2>&1; then
    info "Preflight: release endpoint is reachable"
    return 0
  fi
  warn "install.preflight.release_endpoint_unreachable: $url did not respond; the installer will fall back to the loader-capable source build if the download fails"
  return 0
}

install_binary() {
  local source_binary="$1" destination_binary="$2"
  if [ "$SYSTEM" -eq 1 ]; then
    sudo install -m 0755 "$source_binary" "$destination_binary"
  else
    install -m 0755 "$source_binary" "$destination_binary"
  fi
}

provision_default_semantic_models() {
  local staged_binary="$1"

  info "Provisioning the registered semantic model artifacts..."
  if ! "$staged_binary" download-models; then
    err "Semantic model provisioning failed. The existing fsfs installation was not replaced."
    return 1
  fi

  if ! "$staged_binary" download-models --verify; then
    err "Semantic model verification failed. The existing fsfs installation was not replaced."
    return 1
  fi

  ok "Registered semantic model artifacts are present and verified."
}

verify_staged_binary() {
  local staged_binary="$1" version_output=""

  if version_output="$("$staged_binary" version 2>&1)" && [ -n "$version_output" ]; then
    ok "Staged binary verification passed: $version_output"
    return 0
  fi

  if version_output="$("$staged_binary" --version 2>&1)" && [ -n "$version_output" ]; then
    ok "Staged binary verification passed (--version): $version_output"
    return 0
  fi

  err "Staged binary failed version checks. The existing fsfs installation was not replaced."
  return 1
}

usage() {
  cat <<EOFU
Usage: install.sh [--version vX.Y.Z] [--dest DIR] [--system] [--easy-mode] [--verify] \\
                  [--artifact-url URL] [--checksum HEX] [--checksum-url URL] \\
                  [--force] [--offline] [--from-source] [--lite] [--quiet] [--no-gum]

Options:
  --version vX.Y.Z   Install specific version (default: latest)
  --dest DIR         Install to DIR (default: ~/.local/bin)
  --system           Install to /usr/local/bin (requires sudo)
  --easy-mode        Auto-update PATH in shell rc files
  --verify           Run self-test after install
  --artifact-url URL Install from an explicit archive URL or local path
  --checksum HEX     Expected SHA-256 of the archive (64 hex characters)
  --checksum-url URL Fetch the expected SHA-256 from URL instead of the release
  --force            Reinstall even when the resolved version is already present
  --offline          Never touch the network; requires --version plus a local
                     --artifact-url and --checksum
  --from-source      Build from source instead of downloading binary
  --lite             Force the model-free source profile (~15MB).
                     Implies --from-source; the Cargo default is loader-capable.
  --quiet            Suppress non-error output
  --no-gum           Disable gum formatting even if available
EOFU
}

# An unrecognized flag is rejected rather than ignored: silently dropping a
# mistyped --lite would install a different profile than the operator asked for.
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --version) [ "$#" -ge 2 ] || { err "--version requires a value"; return 2; }; VERSION="$2"; shift 2;;
      --dest) [ "$#" -ge 2 ] || { err "--dest requires a value"; return 2; }; DEST="$2"; shift 2;;
      --system) SYSTEM=1; DEST="/usr/local/bin"; shift;;
      --easy-mode) EASY=1; shift;;
      --verify) VERIFY=1; shift;;
      --artifact-url) [ "$#" -ge 2 ] || { err "--artifact-url requires a value"; return 2; }; ARTIFACT_URL="$2"; shift 2;;
      --checksum) [ "$#" -ge 2 ] || { err "--checksum requires a value"; return 2; }; CHECKSUM="$2"; shift 2;;
      --checksum-url) [ "$#" -ge 2 ] || { err "--checksum-url requires a value"; return 2; }; CHECKSUM_URL="$2"; shift 2;;
      --force) FORCE=1; shift;;
      --offline) OFFLINE=1; shift;;
      --from-source) FROM_SOURCE=1; shift;;
      --lite) LITE=1; FROM_SOURCE=1; shift;;
      --quiet|-q) QUIET=1; shift;;
      --no-gum) NO_GUM=1; shift;;
      -h|--help) usage; return 10;;
      *)
        err "Unknown installer argument: $1"
        err "Run install.sh --help for the supported flags. Nothing was installed or replaced."
        return 2
        ;;
    esac
  done
}

run_installer_contract_test() {
  local action="${1:-}"
  case "$action" in
    checksum)
      [ "$#" -ge 3 ] || { err "contract checksum requires ARCHIVE EXPECTED [TOOL_MODE]"; return 2; }
      verify_archive_checksum "$2" "$3" "${4:-auto}"
      ;;
    manifest)
      [ "$#" -eq 3 ] || { err "contract manifest requires MANIFEST ARTIFACT"; return 2; }
      local resolved
      resolved=$(checksum_from_manifest "$2" "$3")
      validate_sha256 "$resolved" || {
        err "Checksum for $3 is absent from the release manifest"
        return 1
      }
      printf '%s\n' "$resolved"
      ;;
    sidecar)
      [ "$#" -eq 2 ] || { err "contract sidecar requires SIDECAR"; return 2; }
      local resolved
      resolved=$(checksum_from_sidecar "$2")
      validate_sha256 "$resolved" || {
        err "Checksum sidecar is absent or malformed: $2"
        return 1
      }
      printf '%s\n' "$resolved"
      ;;
    route)
      [ "$#" -eq 3 ] || [ "$#" -eq 4 ] || {
        err "contract route requires EXPLICIT_LITE FULL_ARTIFACT_AVAILABLE [TARGET]"
        return 2
      }
      install_route "$2" "$3" "${4:-}"
      ;;
    unsupported)
      [ "$#" -eq 2 ] || { err "contract unsupported requires TARGET"; return 2; }
      fail_unsupported_semantic_platform "$2"
      ;;
    provision)
      [ "$#" -eq 2 ] || { err "contract provision requires STAGED_BINARY"; return 2; }
      provision_default_semantic_models "$2"
      ;;
    verify-staged)
      [ "$#" -eq 2 ] || { err "contract verify-staged requires STAGED_BINARY"; return 2; }
      verify_staged_binary "$2"
      ;;
    output-mode)
      [ "$#" -eq 2 ] || { err "contract output-mode requires QUIET"; return 2; }
      QUIET="$2"
      HAS_GUM=0
      NO_GUM=1
      info "info-output"
      ok "ok-output"
      warn "warn-output"
      err "error-output"
      ;;
    install-built)
      [ "$#" -eq 3 ] || { err "contract install-built requires SOURCE DESTINATION"; return 2; }
      SYSTEM=0
      install_binary "$2" "$3"
      ;;
    args)
      shift
      parse_args "$@" || return $?
      printf 'version=%s dest=%s system=%s easy=%s verify=%s force=%s offline=%s from_source=%s lite=%s quiet=%s no_gum=%s artifact_url=%s checksum=%s\n' \
        "$VERSION" "$DEST" "$SYSTEM" "$EASY" "$VERIFY" "$FORCE" "$OFFLINE" \
        "$FROM_SOURCE" "$LITE" "$QUIET" "$NO_GUM" "$ARTIFACT_URL" "$CHECKSUM"
      ;;
    disk-floor)
      [ "$#" -eq 4 ] || { err "contract disk-floor requires PATH FLOOR_MB LABEL"; return 2; }
      check_disk_floor "$2" "$3" "$4"
      ;;
    dest-writable)
      [ "$#" -eq 2 ] || [ "$#" -eq 3 ] || { err "contract dest-writable requires DEST [SYSTEM]"; return 2; }
      SYSTEM="${3:-0}"
      check_dest_writable "$2"
      ;;
    existing-install)
      [ "$#" -eq 3 ] || { err "contract existing-install requires DEST_BINARY VERSION"; return 2; }
      detect_existing_install "$2" "$3"
      ;;
    offline-preconditions)
      [ "$#" -eq 6 ] || {
        err "contract offline-preconditions requires OFFLINE VERSION FROM_SOURCE ARTIFACT_URL CHECKSUM"
        return 2
      }
      check_offline_preconditions "$2" "$3" "$4" "$5" "$6"
      ;;
    *)
      err "Unknown installer contract test: $action"
      return 2
      ;;
  esac
}

# This non-networked test seam exercises the same routing and checksum
# functions as production without acquiring the installer lock or writing to
# the destination. It is intentionally unavailable unless the checker opts in.
if [ "${FSFS_INSTALL_CONTRACT_TEST:-0}" = "1" ]; then
  run_installer_contract_test "$@"
  exit $?
fi

run_with_spinner() {
  local title="$1"
  shift
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ] && [ "$NO_COLOR_MODE" -eq 0 ] && [ "$QUIET" -eq 0 ]; then
    gum spin --spinner dot --title "$title" -- "$@"
  else
    info "$title"
    "$@"
  fi
}

resolve_version() {
  if [ -n "$VERSION" ]; then return 0; fi

  info "Resolving latest version..."
  local latest_url="https://api.github.com/repos/${OWNER}/${REPO}/releases/latest"
  local tag
  if ! tag=$(curl -fsSL --connect-timeout 30 --max-time 60 -H "Accept: application/vnd.github.v3+json" "$latest_url" 2>/dev/null | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'); then
    tag=""
  fi

  if [ -n "$tag" ]; then
    VERSION="$tag"
    info "Resolved latest version: $VERSION"
  else
    # Try redirect-based resolution as fallback
    local redirect_url="https://github.com/${OWNER}/${REPO}/releases/latest"
    if tag=$(curl -fsSL --connect-timeout 30 --max-time 60 -o /dev/null -w '%{url_effective}' "$redirect_url" 2>/dev/null | sed -E 's|.*/tag/||'); then
      if [ -n "$tag" ] && [[ "$tag" =~ ^v[0-9] ]] && [[ "$tag" != *"/"* ]]; then
        VERSION="$tag"
        info "Resolved latest version via redirect: $VERSION"
        return 0
      fi
    fi
    err "Could not resolve latest version. Use --version vX.Y.Z"
    exit 1
  fi
}

maybe_add_path() {
  case ":$PATH:" in
    *:"$DEST":*) return 0;;
    *)
      if [ "$EASY" -eq 1 ]; then
        local UPDATED=0
        for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
          if [ -e "$rc" ] && [ -w "$rc" ]; then
            if ! grep -qF "$DEST" "$rc" 2>/dev/null; then
              # shellcheck disable=SC2016
              printf '\nexport PATH="%s:$PATH"\n' "$DEST" >> "$rc"
            fi
            UPDATED=1
          fi
        done
        if [ "$UPDATED" -eq 1 ]; then
          warn "PATH updated in shell config; restart your shell to use ${BINARY_NAME}"
        else
          warn "Add $DEST to PATH to use ${BINARY_NAME}"
        fi
      else
        warn "Add $DEST to PATH to use ${BINARY_NAME}"
      fi
    ;;
  esac
}

ensure_rust() {
  if [ "${RUSTUP_INIT_SKIP:-0}" != "0" ]; then
    info "Skipping rustup install (RUSTUP_INIT_SKIP set)"
    return 0
  fi
  if command -v cargo >/dev/null 2>&1 && rustc --version 2>/dev/null | grep -q nightly; then return 0; fi
  if [ "$EASY" -ne 1 ]; then
    if [ -t 0 ]; then
      printf "Install Rust nightly via rustup? (y/N): "
      read -r ans
      case "$ans" in y|Y) :;; *) warn "Skipping rustup install"; return 0;; esac
    fi
  fi
  info "Installing rustup (nightly)"
  curl -fsSL --connect-timeout 30 --max-time 300 https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
  rustup component add rustfmt clippy || true
}

PARSE_STATUS=0
parse_args "$@" || PARSE_STATUS=$?
case "$PARSE_STATUS" in
  0) ;;
  10) exit 0;;             # --help
  *) exit "$PARSE_STATUS";;
esac

# Show header
if [ "$QUIET" -eq 0 ]; then
  if [ "$NO_COLOR_MODE" -eq 1 ]; then
    printf '\nfsfs installer\nTwo-tier hybrid local search\n\n'
  elif [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style \
      --border rounded \
      --border-foreground 39 \
      --padding "0 2" \
      --margin "1 0" \
      "$(gum style --foreground 42 --bold '⚡ fsfs installer')" \
      "$(gum style --foreground 245 'Two-tier hybrid local search (frankensearch)')"
  else
    echo ""
    echo -e "  \033[1;36m╭─────────────────────────────────────────╮\033[0m"
    echo -e "  \033[1;36m│\033[0m  \033[1;32m⚡ fsfs installer\033[0m                       \033[1;36m│\033[0m"
    echo -e "  \033[1;36m│\033[0m  \033[0;90mTwo-tier hybrid local search\033[0m            \033[1;36m│\033[0m"
    echo -e "  \033[1;36m╰─────────────────────────────────────────╯\033[0m"
    echo ""
  fi
fi

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  arm64|aarch64) ARCH="aarch64" ;;
  *) warn "Unknown arch $ARCH, using as-is" ;;
esac

TARGET=""
EXT=""
case "${OS}-${ARCH}" in
  linux-x86_64)   TARGET="x86_64-unknown-linux-musl"; EXT="tar.xz" ;;
  linux-aarch64)  TARGET="aarch64-unknown-linux-musl"; EXT="tar.xz" ;;
  darwin-x86_64)  TARGET="x86_64-apple-darwin"; EXT="tar.xz" ;;
  darwin-aarch64) TARGET="aarch64-apple-darwin"; EXT="tar.xz" ;;
  *) :;;
esac

if [ "$LITE" -eq 0 ] && [ "$TARGET" = "x86_64-apple-darwin" ]; then
  fail_unsupported_semantic_platform "$TARGET"
  exit $?
fi

# Offline inputs are validated before anything reaches for the network.
check_offline_preconditions "$OFFLINE" "$VERSION" "$FROM_SOURCE" "$ARTIFACT_URL" "$CHECKSUM" || exit 1

# Version lookup may use the network. Unsupported ordinary profiles are
# rejected above before any release lookup, artifact probe, or filesystem
# installation mutation. Intel macOS remains available only through --lite.
resolve_version

# Remaining preflight. Every check below is observational, so a rejection here
# leaves any existing installation exactly as it was.
check_dest_writable "$DEST" || exit 1
check_disk_floor "$DEST" "$MIN_DEST_MB" "install destination" || exit 1

STAGE_ROOT="${TMPDIR:-/tmp}"
if [ "$FROM_SOURCE" -eq 1 ] || [ -z "$TARGET" ]; then
  check_disk_floor "$STAGE_ROOT" "$MIN_STAGE_SOURCE_MB" "source build staging area" || exit 1
else
  check_disk_floor "$STAGE_ROOT" "$MIN_STAGE_ARTIFACT_MB" "artifact staging area" || exit 1
fi

EXISTING_STATE=$(detect_existing_install "$DEST/${BINARY_NAME}" "$VERSION")
case "$EXISTING_STATE" in
  fresh)
    info "Preflight: no existing ${BINARY_NAME} at $DEST"
    ;;
  same-version)
    if [ "$FORCE" -eq 0 ]; then
      ok "${BINARY_NAME} ${VERSION} is already installed at $DEST/${BINARY_NAME}; pass --force to reinstall"
      exit 0
    fi
    info "Preflight: ${VERSION} already installed; --force requested, reinstalling"
    ;;
  different-version)
    info "Preflight: upgrading the existing ${BINARY_NAME} at $DEST to ${VERSION}"
    ;;
  *)
    warn "An existing $DEST/${BINARY_NAME} did not report a version; it will be replaced only after the new binary verifies"
    ;;
esac

if [ "$FROM_SOURCE" -eq 0 ]; then
  case "$ARTIFACT_URL" in
    http://*|https://*) check_release_endpoint_reachable "$ARTIFACT_URL" ;;
    '') check_release_endpoint_reachable "https://github.com/${OWNER}/${REPO}/releases" ;;
    *) info "Preflight: installing from the local artifact $ARTIFACT_URL" ;;
  esac
fi

mkdir -p "$DEST"

# Build artifact filename and download URL.
# dsr artifact naming: fsfs-${version_bare}-${target_triple}.${ext}
VERSION_BARE="${VERSION#v}"  # strip leading v for artifact naming
TAR=""
URL=""
if [ "$FROM_SOURCE" -eq 0 ]; then
  if [ -n "$ARTIFACT_URL" ]; then
    TAR=$(basename "$ARTIFACT_URL")
    URL="$ARTIFACT_URL"
  elif [ -n "$TARGET" ]; then
    TAR="${BINARY_NAME}-${VERSION_BARE}-${TARGET}.${EXT}"
    URL="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${TAR}"
  else
    warn "No prebuilt artifact for ${OS}/${ARCH}; falling back to build-from-source"
    FROM_SOURCE=1
  fi
fi

# Cross-platform locking using mkdir (atomic on all POSIX systems including macOS)
LOCK_DIR="${LOCK_FILE}.d"
LOCKED=0
if mkdir "$LOCK_DIR" 2>/dev/null; then
  LOCKED=1
  echo $$ > "$LOCK_DIR/pid"
else
  if [ -f "$LOCK_DIR/pid" ]; then
    OLD_PID=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && ! kill -0 "$OLD_PID" 2>/dev/null; then
      rm -rf "$LOCK_DIR"
      if mkdir "$LOCK_DIR" 2>/dev/null; then
        LOCKED=1
        echo $$ > "$LOCK_DIR/pid"
      fi
    fi
  fi
  if [ "$LOCKED" -eq 0 ]; then
    err "Another installer is running (lock $LOCK_DIR)"
    exit 1
  fi
fi

cleanup() {
  rm -rf "$TMP"
  if [ "$LOCKED" -eq 1 ]; then rm -rf "$LOCK_DIR"; fi
}

TMP=$(mktemp -d)
trap cleanup EXIT

download_with_progress() {
  local url="$1" dest="$2" label="${3:-Downloading}"
  local size_bytes="" size_human=""

  # A local path or file:// URL is copied rather than fetched, which is what
  # makes --offline installs possible from a pre-staged archive.
  case "$url" in
    file://*)
      local local_path="${url#file://}"
      if [ ! -f "$local_path" ]; then
        err "Local artifact not found: $local_path"
        return 1
      fi
      info "$label (local artifact)"
      cp -- "$local_path" "$dest"
      return 0
      ;;
    *://*) : ;;
    *)
      if [ ! -f "$url" ]; then
        err "Local artifact not found: $url"
        return 1
      fi
      info "$label (local artifact)"
      cp -- "$url" "$dest"
      return 0
      ;;
  esac

  # Probe content-length for a helpful pre-download message
  if size_bytes=$(curl -fsSL --connect-timeout 10 --max-time 15 -I "$url" 2>/dev/null \
        | grep -i '^content-length:' | awk '{print $2}' | tr -d '\r'); then
    if [ -n "$size_bytes" ] && [ "$size_bytes" -gt 0 ] 2>/dev/null; then
      if [ "$size_bytes" -ge 1073741824 ]; then
        size_human="$(awk "BEGIN{printf \"%.1f GB\", $size_bytes/1073741824}")"
      elif [ "$size_bytes" -ge 1048576 ]; then
        size_human="$(awk "BEGIN{printf \"%.0f MB\", $size_bytes/1048576}")"
      else
        size_human="$(awk "BEGIN{printf \"%.0f KB\", $size_bytes/1024}")"
      fi
    fi
  fi

  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ] && [ "$NO_COLOR_MODE" -eq 0 ] && [ "$QUIET" -eq 0 ]; then
    # ── gum: rich styled output ──
    if [ -n "$size_human" ]; then
      gum style --foreground 39 "$(printf '↓ %s  %s  (%s)' "$label" "$(gum style --faint --italic "$(basename "$url")")" \
        "$(gum style --bold --foreground 213 "$size_human")")"
    else
      gum style --foreground 39 "↓ ${label}"
    fi
    # Use gum spin wrapping curl progress (curl still writes its bar to stderr)
    if ! curl -fL --progress-bar --connect-timeout 30 --max-time 1800 "$url" -o "$dest"; then
      return 1
    fi
  elif [ -t 1 ] && [ "$NO_COLOR_MODE" -eq 0 ] && [ "$QUIET" -eq 0 ]; then
    # ── Interactive terminal: styled ANSI progress ──
    if [ -n "$size_human" ]; then
      printf '\033[1;36m↓\033[0m %s \033[2m%s\033[0m  \033[1;35m%s\033[0m\n' \
        "$label" "$(basename "$url")" "$size_human"
    else
      printf '\033[1;36m↓\033[0m %s \033[2m%s\033[0m\n' "$label" "$(basename "$url")"
    fi
    if ! curl -fL --progress-bar --connect-timeout 30 --max-time 1800 "$url" -o "$dest" 2>&1; then
      return 1
    fi
  else
    # ── Non-interactive / quiet: silent download ──
    info "$label"
    if ! curl -fsSL --connect-timeout 30 --max-time 1800 "$url" -o "$dest"; then
      return 1
    fi
  fi
  return 0
}

if [ "$FROM_SOURCE" -eq 0 ]; then
  if ! download_with_progress "$URL" "$TMP/$TAR" "Downloading ${BINARY_NAME} ${VERSION}"; then
    if [ "$OFFLINE" -eq 1 ]; then
      err "The offline artifact could not be staged and --offline forbids the source-build fallback."
      err "The existing fsfs installation was not replaced."
      exit 1
    fi
    ROUTE=$(install_route "$LITE" 0 "$TARGET")
    case "$ROUTE" in
      source-default)
        warn "Full artifact download failed; building the loader-capable default from source"
        FROM_SOURCE=1
        ;;
      unsupported-semantic)
        fail_unsupported_semantic_platform "$TARGET"
        exit $?
        ;;
      *)
        err "Internal installer routing error: expected source-default or unsupported-semantic, got $ROUTE"
        exit 1
        ;;
    esac
  fi
fi

if [ "$FROM_SOURCE" -eq 1 ]; then
  info "Building from source (requires git, rust nightly)"
  ensure_rust
  if [ -n "$VERSION" ]; then
    git clone --depth 1 --recurse-submodules --branch "$VERSION" "https://github.com/${OWNER}/${REPO}.git" "$TMP/src"
  else
    git clone --depth 1 --recurse-submodules "https://github.com/${OWNER}/${REPO}.git" "$TMP/src"
  fi
  # Remove optional workspace members whose path dependencies (e.g. fast_cmaes)
  # live outside the repository and are unavailable in a fresh clone.
  if [ -f "$TMP/src/Cargo.toml" ]; then
    sed -i.bak '/"tools\/optimize_params"/d' "$TMP/src/Cargo.toml"
    rm -f "$TMP/src/Cargo.toml.bak"
  fi
  # Unset env vars that would redirect cargo output away from the default
  # target directory.  Without this, users with CARGO_TARGET_DIR or
  # CARGO_BUILD_TARGET set (common among Rust developers) would see a
  # successful compile followed by a spurious "Build failed" because the
  # binary lands in an unexpected location.
  if [ "$LITE" -eq 1 ]; then
    info "Building lite variant (semantic model loaders disabled)"
    (cd "$TMP/src" && unset CARGO_TARGET_DIR CARGO_BUILD_TARGET_DIR CARGO_BUILD_TARGET && cargo build --release -p frankensearch-fsfs --no-default-features)
  else
    info "Building default semantic-loader variant (model bytes acquired separately)"
    (cd "$TMP/src" && unset CARGO_TARGET_DIR CARGO_BUILD_TARGET_DIR CARGO_BUILD_TARGET && cargo build --release -p frankensearch-fsfs)
  fi
  BIN="$TMP/src/target/release/${BINARY_NAME}"
  if [ ! -x "$BIN" ]; then
    # Fallback: search for the binary in case a .cargo/config.toml or other
    # mechanism placed it elsewhere under the source tree.
    FOUND_BIN=$(find "$TMP/src/target" -maxdepth 4 -type f -name "${BINARY_NAME}" -perm -111 2>/dev/null | head -n 1)
    if [ -n "$FOUND_BIN" ] && [ -x "$FOUND_BIN" ]; then
      warn "Binary was not at expected path ($BIN), found at $FOUND_BIN"
      BIN="$FOUND_BIN"
    else
      err "Build succeeded but binary not found at $BIN"
      err "Check CARGO_TARGET_DIR or .cargo/config.toml target-dir settings"
      exit 1
    fi
  fi
  if [ "$LITE" -eq 0 ]; then
    if ! provision_default_semantic_models "$BIN"; then
      exit 1
    fi
  fi
  if [ "$VERIFY" -eq 1 ]; then
    if ! verify_staged_binary "$BIN"; then
      exit 1
    fi
  fi

  install_binary "$BIN" "$DEST/${BINARY_NAME}"
  ok "Installed to $DEST/${BINARY_NAME} (source build)"
  maybe_add_path
  if [ "$VERIFY" -eq 1 ]; then
    if ! SELF_TEST_OUTPUT=$("$DEST/${BINARY_NAME}" version 2>&1); then
      err "Self-test failed: $SELF_TEST_OUTPUT"
      exit 1
    fi
    ok "Self-test complete: $SELF_TEST_OUTPUT"
  fi
  if [ "$LITE" -eq 1 ]; then
    info "Model-free build: Model2Vec and FastEmbed execution are not compiled."
    info "Install the standard build for semantic retrieval; downloaded files alone cannot activate this lite binary."
  else
    info "Semantic loaders installed. Provision and verify the registered models with:"
    info "  ${BINARY_NAME} download-models potion-multilingual-128m"
    info "  ${BINARY_NAME} download-models all-minilm-l6-v2"
    info "  ${BINARY_NAME} download-models potion-multilingual-128m --verify"
    info "  ${BINARY_NAME} download-models all-minilm-l6-v2 --verify"
    info "  ${BINARY_NAME} index /path/to/files"
    info "  ${BINARY_NAME} search \"your query\""
  fi
  ok "Done. Binary at: $DEST/${BINARY_NAME}"
  exit 0
fi

# Verify checksum
if [ -z "$CHECKSUM" ]; then
  CHECKSUM_URL_DEFAULTED=0
  if [ -z "$CHECKSUM_URL" ]; then
    CHECKSUM_URL="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/SHA256SUMS"
    CHECKSUM_URL_DEFAULTED=1
  fi
  info "Fetching checksum from ${CHECKSUM_URL}"
  CHECKSUM_FILE="$TMP/SHA256SUMS"
  if curl -fsSL --connect-timeout 30 --max-time 60 "$CHECKSUM_URL" -o "$CHECKSUM_FILE"; then
    CHECKSUM=$(checksum_from_manifest "$CHECKSUM_FILE" "$TAR" || true)
  fi
  if [ -z "$CHECKSUM" ] && [ "$CHECKSUM_URL_DEFAULTED" -eq 1 ]; then
    SIDECAR_CHECKSUM_URL="${URL}.sha256"
    info "Fetching checksum sidecar from ${SIDECAR_CHECKSUM_URL}"
    if curl -fsSL --connect-timeout 30 --max-time 60 "$SIDECAR_CHECKSUM_URL" -o "$CHECKSUM_FILE"; then
      CHECKSUM=$(checksum_from_sidecar "$CHECKSUM_FILE" || true)
    fi
  fi
  if [ -z "$CHECKSUM" ]; then
    err "Checksum for ${TAR} is unavailable; refusing to install an unverified artifact"
    exit 1
  fi
fi

verify_archive_checksum "$TMP/$TAR" "$CHECKSUM" || exit 1
ok "Checksum verified"

# Extract
info "Extracting"
case "$TAR" in
  *.tar.xz)  tar -xJf "$TMP/$TAR" -C "$TMP" ;;
  *.tar.gz)  tar -xzf "$TMP/$TAR" -C "$TMP" ;;
  *.zip)     unzip -qo "$TMP/$TAR" -d "$TMP" ;;
  *)         err "Unknown archive format: $TAR"; exit 1 ;;
esac

# Find the binary in extracted files
BIN="$TMP/${BINARY_NAME}"
if [ ! -x "$BIN" ]; then
  BIN=$(find "$TMP" -maxdepth 3 -type f -name "${BINARY_NAME}" -perm -111 2>/dev/null | head -n 1)
fi
[ -x "$BIN" ] || { err "Binary not found in archive"; exit 1; }

if [ "$LITE" -eq 0 ]; then
  if ! provision_default_semantic_models "$BIN"; then
    exit 1
  fi
fi

if [ "$VERIFY" -eq 1 ]; then
  if ! verify_staged_binary "$BIN"; then
    exit 1
  fi
fi

install_binary "$BIN" "$DEST/${BINARY_NAME}"
ok "Installed to $DEST/${BINARY_NAME}"
maybe_add_path

if [ "$VERIFY" -eq 1 ]; then
  if ! SELF_TEST_OUTPUT=$("$DEST/${BINARY_NAME}" version 2>&1); then
    err "Self-test failed: $SELF_TEST_OUTPUT"
    exit 1
  fi
  ok "Self-test complete: $SELF_TEST_OUTPUT"
fi

if [ "$QUIET" -eq 0 ]; then
if [ "$NO_COLOR_MODE" -eq 1 ]; then
  printf '\nInstallation complete!\n'
  printf 'Binary: %s\n' "$DEST/${BINARY_NAME}"
  printf 'Version: %s\n\n' "$VERSION"
  printf '%s\n' 'Quick start:'
  printf '%s\n' '  fsfs download-models potion-multilingual-128m'
  printf '%s\n' '  fsfs download-models all-minilm-l6-v2'
  printf '%s\n' '  fsfs download-models potion-multilingual-128m --verify'
  printf '%s\n' '  fsfs download-models all-minilm-l6-v2 --verify'
  printf '%s\n' '  fsfs index /path/to/files   Index a directory'
  printf '%s\n' '  fsfs search "your query"    Search your index'
  printf '%s\n\n' '  fsfs                        Interactive TUI'
elif [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
  echo ""
  gum style \
    --border rounded \
    --border-foreground 42 \
    --padding "0 2" \
    --margin "0" \
    "$(gum style --foreground 42 --bold '✓ Installation complete!')" \
    "" \
    "$(gum style --foreground 245 "Binary:  $(gum style --bold "$DEST/${BINARY_NAME}")")" \
    "$(gum style --foreground 245 "Version: $(gum style --bold "${VERSION}")")" \
    "" \
    "$(gum style --foreground 39 --bold 'Quick start:')" \
    "$(gum style --foreground 245 '  fsfs download-models potion-multilingual-128m')" \
    "$(gum style --foreground 245 '  fsfs download-models all-minilm-l6-v2')" \
    "$(gum style --foreground 245 '  fsfs download-models potion-multilingual-128m --verify')" \
    "$(gum style --foreground 245 '  fsfs download-models all-minilm-l6-v2 --verify')" \
    "$(gum style --foreground 245 '  fsfs index /path/to/files   Index a directory')" \
    "$(gum style --foreground 245 '  fsfs search "your query"    Search your index')" \
    "$(gum style --foreground 245 '  fsfs                        Interactive TUI')"
  echo ""
else
  echo ""
  echo -e "  \033[1;32m╭─────────────────────────────────────────╮\033[0m"
  echo -e "  \033[1;32m│\033[0m  \033[1;32m✓ Installation complete!\033[0m                 \033[1;32m│\033[0m"
  echo -e "  \033[1;32m│\033[0m                                         \033[1;32m│\033[0m"
  BINARY_LINE="  Binary:  $DEST/${BINARY_NAME}"
  VERSION_LINE="  Version: ${VERSION}"
  BOX_WIDTH=41
  BPAD=$(( BOX_WIDTH - ${#BINARY_LINE} ))
  VPAD=$(( BOX_WIDTH - ${#VERSION_LINE} ))
  [ "$BPAD" -lt 1 ] && BPAD=1
  [ "$VPAD" -lt 1 ] && VPAD=1
  echo -e "  \033[1;32m│\033[0m  Binary:  \033[1m$DEST/${BINARY_NAME}\033[0m$(printf '%*s' "$BPAD" '')\033[1;32m│\033[0m"
  echo -e "  \033[1;32m│\033[0m  Version: \033[1m${VERSION}\033[0m$(printf '%*s' "$VPAD" '')\033[1;32m│\033[0m"
  echo -e "  \033[1;32m│\033[0m                                         \033[1;32m│\033[0m"
  echo -e "  \033[1;32m│\033[0m  \033[1;36mQuick start:\033[0m                          \033[1;32m│\033[0m"
  echo -e "  \033[1;32m│\033[0m  \033[0;90m$ fsfs download-models potion-multilingual-128m\033[0m"
  echo -e "  \033[1;32m│\033[0m  \033[0;90m$ fsfs download-models all-minilm-l6-v2\033[0m"
  echo -e "  \033[1;32m│\033[0m  \033[0;90m$ fsfs download-models potion-multilingual-128m --verify\033[0m"
  echo -e "  \033[1;32m│\033[0m  \033[0;90m$ fsfs download-models all-minilm-l6-v2 --verify\033[0m"
  echo -e "  \033[1;32m│\033[0m  \033[0;90m$ fsfs index /path/to/files\033[0m           \033[1;32m│\033[0m"
  echo -e "  \033[1;32m│\033[0m  \033[0;90m$ fsfs search \"your query\"\033[0m            \033[1;32m│\033[0m"
  echo -e "  \033[1;32m│\033[0m  \033[0;90m$ fsfs\033[0m  \033[2m(interactive TUI)\033[0m          \033[1;32m│\033[0m"
  echo -e "  \033[1;32m╰─────────────────────────────────────────╯\033[0m"
  echo ""
fi
fi
