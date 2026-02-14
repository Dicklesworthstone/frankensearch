#!/usr/bin/env bats

# Tests for bd-2w7x.18: Prebuilt binary download with SHA-256 verification.
#
# These tests exercise the download, checksum verification, extraction, and
# installation functions in scripts/install.sh using synthetic local artifacts.
# No network access is required.

load "helpers/common.bash"

INSTALLER=""

setup() {
  REPO_ROOT="$(repo_root)"
  INSTALLER="${REPO_ROOT}/scripts/install.sh"
  setup_installer_test_env

  # Create a fake fsfs binary.
  FAKE_BINARY_DIR="${TEST_ROOT}/fake_binary"
  mkdir -p "${FAKE_BINARY_DIR}"
  cat > "${FAKE_BINARY_DIR}/fsfs" << 'SCRIPT'
#!/bin/bash
echo "fsfs version 0.1.0-test"
SCRIPT
  chmod +x "${FAKE_BINARY_DIR}/fsfs"

  # Create the tar.xz archive.
  ARTIFACT_DIR="${TEST_ROOT}/artifacts"
  mkdir -p "${ARTIFACT_DIR}"
  (cd "${FAKE_BINARY_DIR}" && tar -cJf "${ARTIFACT_DIR}/fsfs-x86_64-unknown-linux-musl.tar.xz" fsfs)

  # Compute the correct SHA-256 checksum.
  CORRECT_HASH="$(sha256sum "${ARTIFACT_DIR}/fsfs-x86_64-unknown-linux-musl.tar.xz" | awk '{print $1}')"

  # Write the per-artifact checksum file.
  echo "${CORRECT_HASH}  fsfs-x86_64-unknown-linux-musl.tar.xz" \
    > "${ARTIFACT_DIR}/fsfs-x86_64-unknown-linux-musl.tar.xz.sha256"

  # Write the combined checksums.txt.
  echo "${CORRECT_HASH}  fsfs-x86_64-unknown-linux-musl.tar.xz" \
    > "${ARTIFACT_DIR}/checksums.txt"

  INSTALL_DEST="${TEST_ROOT}/install_dest"
  mkdir -p "${INSTALL_DEST}"
}

# Source the installer in a subshell with overridden state.
# Reads CHECKSUM and VERIFY from exported env vars.
# Note: The eval re-sources the script which resets VERIFY/CHECKSUM to defaults,
# so we save and restore them.
run_installer_fn() {
  local fn_name="$1"
  shift
  local _saved_verify="${VERIFY:-false}"
  local _saved_checksum="${CHECKSUM:-}"
  (
    eval "$(sed '/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d' "${INSTALLER}")"
    TARGET_OS="unknown-linux-musl"
    TARGET_ARCH="x86_64"
    TARGET_TRIPLE="x86_64-unknown-linux-musl"
    TEMP_DIR="$(mktemp -d)"
    DEFAULT_BINARY_NAME="fsfs"
    QUIET=false
    USE_COLOR=false
    HAVE_GUM=false
    VERIFY="${_saved_verify}"
    CHECKSUM="${_saved_checksum}"
    DEST_DIR="${INSTALL_DEST}"

    # Copy test archive into TEMP_DIR to simulate download.
    cp "${ARTIFACT_DIR}/fsfs-x86_64-unknown-linux-musl.tar.xz" "${TEMP_DIR}/" 2>/dev/null || true
    cp "${ARTIFACT_DIR}/fsfs-x86_64-unknown-linux-musl.tar.xz.sha256" "${TEMP_DIR}/" 2>/dev/null || true

    "$fn_name" "$@"
  )
}

# ---------------------------------------------------------------------------
# compute_sha256: produces correct hex digest
# ---------------------------------------------------------------------------

@test "compute_sha256 produces correct hex digest" {
  export CHECKSUM="" VERIFY=false
  run run_installer_fn compute_sha256 \
    "${ARTIFACT_DIR}/fsfs-x86_64-unknown-linux-musl.tar.xz"
  [ "$status" -eq 0 ]
  [[ "$output" == "${CORRECT_HASH}" ]]
}

# ---------------------------------------------------------------------------
# verify_artifact_checksum: correct hash passes
# ---------------------------------------------------------------------------

@test "verify_artifact_checksum passes with correct --checksum" {
  export CHECKSUM="${CORRECT_HASH}" VERIFY=true
  run run_installer_fn verify_artifact_checksum
  [ "$status" -eq 0 ]
  [[ "$output" == *"SHA-256 checksum verified"* ]]
}

# ---------------------------------------------------------------------------
# verify_artifact_checksum: wrong hash fails
# ---------------------------------------------------------------------------

@test "verify_artifact_checksum fails with wrong --checksum" {
  export CHECKSUM="0000000000000000000000000000000000000000000000000000000000000000"
  export VERIFY=true
  run run_installer_fn verify_artifact_checksum
  [ "$status" -ne 0 ]
  [[ "$output" == *"checksum mismatch"* ]]
}

# ---------------------------------------------------------------------------
# verify_artifact_checksum: --verify without checksum fails
# ---------------------------------------------------------------------------

@test "verify_artifact_checksum dies when --verify set but no checksum source" {
  export CHECKSUM="" VERIFY=true
  run run_installer_fn verify_artifact_checksum
  [ "$status" -ne 0 ]
  [[ "$output" == *"Checksum verification was requested"* ]]
}

# ---------------------------------------------------------------------------
# verify_artifact_checksum: no --verify, no checksum => warning + pass
# ---------------------------------------------------------------------------

@test "verify_artifact_checksum warns but passes without --verify or --checksum" {
  export CHECKSUM="" VERIFY=false
  run run_installer_fn verify_artifact_checksum
  [ "$status" -eq 0 ]
  [[ "$output" == *"No checksum available"* ]]
}

# ---------------------------------------------------------------------------
# extract_archive: tar.xz extraction finds the binary
# ---------------------------------------------------------------------------

@test "extract_archive extracts fsfs from tar.xz" {
  export CHECKSUM="" VERIFY=false
  run run_installer_fn extract_archive
  [ "$status" -eq 0 ]
  [[ "$output" == *"Extracted binary: fsfs"* ]]
}

# ---------------------------------------------------------------------------
# install_binary: places binary with correct permissions
# ---------------------------------------------------------------------------

@test "install_binary creates executable binary at destination" {
  (
    eval "$(sed '/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d' "${INSTALLER}")"
    TARGET_OS="unknown-linux-musl"
    TARGET_ARCH="x86_64"
    TARGET_TRIPLE="x86_64-unknown-linux-musl"
    TEMP_DIR="$(mktemp -d)"
    DEFAULT_BINARY_NAME="fsfs"
    QUIET=false
    USE_COLOR=false
    HAVE_GUM=false
    VERIFY=false
    CHECKSUM=""
    DEST_DIR="${INSTALL_DEST}"
    cp "${ARTIFACT_DIR}/fsfs-x86_64-unknown-linux-musl.tar.xz" "${TEMP_DIR}/"
    extract_archive
    install_binary
  )
  [ -x "${INSTALL_DEST}/fsfs" ]
}

# ---------------------------------------------------------------------------
# verify_installation: installed binary produces --version output
# ---------------------------------------------------------------------------

@test "verify_installation succeeds for valid binary" {
  (
    eval "$(sed '/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d' "${INSTALLER}")"
    TARGET_OS="unknown-linux-musl"
    TARGET_ARCH="x86_64"
    TARGET_TRIPLE="x86_64-unknown-linux-musl"
    TEMP_DIR="$(mktemp -d)"
    DEFAULT_BINARY_NAME="fsfs"
    QUIET=false
    USE_COLOR=false
    HAVE_GUM=false
    VERIFY=false
    CHECKSUM=""
    DEST_DIR="${INSTALL_DEST}"
    cp "${ARTIFACT_DIR}/fsfs-x86_64-unknown-linux-musl.tar.xz" "${TEMP_DIR}/"
    extract_archive
    install_binary
    verify_installation
  )
  [ "$?" -eq 0 ]
}

# ---------------------------------------------------------------------------
# Full pipeline: download (simulated) + verify + extract + install
# ---------------------------------------------------------------------------

@test "full pipeline: extract, verify checksum, install, verify binary" {
  run bash -c '
    eval "$(sed '"'"'/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d'"'"' "'"${INSTALLER}"'")"
    TARGET_OS="unknown-linux-musl"
    TARGET_ARCH="x86_64"
    TARGET_TRIPLE="x86_64-unknown-linux-musl"
    TEMP_DIR="$(mktemp -d)"
    DEFAULT_BINARY_NAME="fsfs"
    QUIET=false
    USE_COLOR=false
    HAVE_GUM=false
    VERIFY=true
    CHECKSUM="'"${CORRECT_HASH}"'"
    DEST_DIR="'"${INSTALL_DEST}"'"
    cp "'"${ARTIFACT_DIR}"'/fsfs-x86_64-unknown-linux-musl.tar.xz" "${TEMP_DIR}/"
    verify_artifact_checksum
    extract_archive
    install_binary
    verify_installation
  '
  [ "$status" -eq 0 ]
  [[ "$output" == *"SHA-256 checksum verified"* ]]
  [[ "$output" == *"Extracted binary"* ]]
  [[ "$output" == *"Installed fsfs"* ]]
  [[ "$output" == *"Binary verification passed"* ]]
}

# ---------------------------------------------------------------------------
# artifact_name: generates correct filename
# ---------------------------------------------------------------------------

@test "artifact_name returns correct tar.xz name" {
  export CHECKSUM="" VERIFY=false
  run run_installer_fn artifact_name
  [ "$status" -eq 0 ]
  [[ "$output" == "fsfs-x86_64-unknown-linux-musl.tar.xz" ]]
}

# ---------------------------------------------------------------------------
# checksum_file_name: appends .sha256
# ---------------------------------------------------------------------------

@test "checksum_file_name appends .sha256 to artifact name" {
  export CHECKSUM="" VERIFY=false
  run run_installer_fn checksum_file_name
  [ "$status" -eq 0 ]
  [[ "$output" == "fsfs-x86_64-unknown-linux-musl.tar.xz.sha256" ]]
}

# ---------------------------------------------------------------------------
# Case-insensitive checksum comparison
# ---------------------------------------------------------------------------

@test "verify_artifact_checksum accepts uppercase hex in --checksum" {
  UPPER_HASH="$(echo "${CORRECT_HASH}" | tr '[:lower:]' '[:upper:]')"
  export CHECKSUM="${UPPER_HASH}" VERIFY=true
  run run_installer_fn verify_artifact_checksum
  [ "$status" -eq 0 ]
  [[ "$output" == *"SHA-256 checksum verified"* ]]
}

# ---------------------------------------------------------------------------
# --from-source triggers clear error message
# ---------------------------------------------------------------------------

@test "scripts/install.sh --from-source fails with not-implemented message" {
  run bash "${INSTALLER}" --from-source --offline --version v0.1.0 --dest "${INSTALL_DEST}" --force
  [ "$status" -ne 0 ]
  [[ "$output" == *"not yet implemented"* ]]
}

# ---------------------------------------------------------------------------
# scripts/install.sh --help shows new download-related options
# ---------------------------------------------------------------------------

@test "scripts/install.sh --help includes --verify and --checksum flags" {
  run bash "${INSTALLER}" --help
  [ "$status" -eq 0 ]
  [[ "$output" == *"--verify"* ]]
  [[ "$output" == *"--checksum"* ]]
}

# ---------------------------------------------------------------------------
# Platform detection matrix
# ---------------------------------------------------------------------------

@test "detect_platform resolves Darwin arm64 to aarch64-apple-darwin" {
  run bash -c '
    eval "$(sed '"'"'/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d'"'"' "'"${INSTALLER}"'")"
    ok() { :; }
    uname() {
      case "${1:-}" in
        -s) printf "Darwin\n" ;;
        -m) printf "arm64\n" ;;
        *) command uname "$@" ;;
      esac
    }
    detect_platform
    printf "%s" "${TARGET_TRIPLE}"
  '
  [ "$status" -eq 0 ]
  [[ "$output" == *"aarch64-apple-darwin"* ]]
}

@test "detect_platform fails clearly for unsupported OS" {
  run bash -c '
    eval "$(sed '"'"'/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d'"'"' "'"${INSTALLER}"'")"
    ok() { :; }
    uname() {
      case "${1:-}" in
        -s) printf "FreeBSD\n" ;;
        -m) printf "x86_64\n" ;;
        *) command uname "$@" ;;
      esac
    }
    detect_platform
  '
  [ "$status" -ne 0 ]
  [[ "$output" == *"Unsupported operating system"* ]]
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

@test "check_network_connectivity skips downloader requirements in offline mode" {
  run bash -c '
    eval "$(sed '"'"'/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d'"'"' "'"${INSTALLER}"'")"
    OFFLINE=true
    has_cmd() { return 1; }
    check_network_connectivity
  '
  [ "$status" -eq 0 ]
}

@test "check_network_connectivity fails in online mode without curl or wget" {
  run bash -c '
    eval "$(sed '"'"'/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d'"'"' "'"${INSTALLER}"'")"
    OFFLINE=false
    has_cmd() { return 1; }
    check_network_connectivity
  '
  [ "$status" -ne 0 ]
  [[ "$output" == *"Need curl or wget for connectivity checks"* ]]
}

@test "check_write_permissions fails when destination parent directory is missing" {
  run bash -c '
    eval "$(sed '"'"'/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d'"'"' "'"${INSTALLER}"'")"
    DEST_DIR="'"${TEST_ROOT}"'/missing-parent/subdir/bin"
    check_write_permissions
  '
  [ "$status" -ne 0 ]
  [[ "$output" == *"Destination parent directory does not exist"* ]]
}

@test "check_disk_space fails with clear budget message when free space is too low" {
  run bash -c '
    eval "$(sed '"'"'/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d'"'"' "'"${INSTALLER}"'")"
    DEST_DIR="'"${INSTALL_DEST}"'"
    df() {
      printf "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
      printf "mockfs 100000 99950 50 99%% /tmp\n"
    }
    check_disk_space
  '
  [ "$status" -ne 0 ]
  [[ "$output" == *"At least 200MB free disk space is required"* ]]
}

# ---------------------------------------------------------------------------
# Download/checksum behavior
# ---------------------------------------------------------------------------

@test "resolve_expected_checksum falls back to checksums.txt when .sha256 is unavailable" {
  run bash -c '
    eval "$(sed '"'"'/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d'"'"' "'"${INSTALLER}"'")"
    TARGET_OS="unknown-linux-musl"
    TARGET_ARCH="x86_64"
    TARGET_TRIPLE="x86_64-unknown-linux-musl"
    TEMP_DIR="$(mktemp -d)"
    CHECKSUM=""
    cp "'"${ARTIFACT_DIR}"'/fsfs-x86_64-unknown-linux-musl.tar.xz" "${TEMP_DIR}/"
    http_download() {
      local url="$1"
      local output="$2"
      case "${url}" in
        *".sha256") return 1 ;;
        *"checksums.txt") cp "'"${ARTIFACT_DIR}"'/checksums.txt" "${output}" ;;
        *) return 1 ;;
      esac
    }
    resolve_expected_checksum
  '
  [ "$status" -eq 0 ]
  [[ "$output" == *"${CORRECT_HASH}"* ]]
}

@test "http_download uses curl retry and backoff flags" {
  run bash -c '
    eval "$(sed '"'"'/^main "\$@"/d; /^if \[\[.*BASH_SOURCE/d'"'"' "'"${INSTALLER}"'")"
    has_cmd() { [[ "${1:-}" == "curl" ]]; }
    curl() {
      printf "%s\n" "$*" > "'"${TEST_ROOT}"'/curl-args.txt"
      return 0
    }
    http_download "https://example.invalid/fsfs.tar.xz" "'"${TEST_ROOT}"'/out.tar.xz"
    grep -F -- "--retry 3" "'"${TEST_ROOT}"'/curl-args.txt" >/dev/null
    grep -F -- "--retry-delay 2" "'"${TEST_ROOT}"'/curl-args.txt" >/dev/null
    grep -F -- "--max-time 120" "'"${TEST_ROOT}"'/curl-args.txt" >/dev/null
  '
  [ "$status" -eq 0 ]
}
