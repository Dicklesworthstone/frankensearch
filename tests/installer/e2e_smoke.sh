#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TMP_ROOT="${FSFS_E2E_TMPDIR:-$(mktemp -d)}"
KEEP_TMP="${FSFS_E2E_KEEP_TMPDIR:-0}"
INSTALL_MODE="${FSFS_E2E_INSTALL_MODE:-local}"
LOG_DIR="${TMP_ROOT}/logs"
PROJECT_DIR="${TMP_ROOT}/project"
INDEX_DIR="${PROJECT_DIR}/.frankensearch"
INSTALL_BIN_DIR="${TMP_ROOT}/install/bin"
FSFS_BIN="${INSTALL_BIN_DIR}/fsfs"

export HOME="${TMP_ROOT}/home"
export XDG_CONFIG_HOME="${HOME}/.config"
export XDG_CACHE_HOME="${HOME}/.cache"
export XDG_DATA_HOME="${HOME}/.local/share"
export FRANKENSEARCH_MODEL_DIR="${XDG_DATA_HOME}/frankensearch/models"
export FRANKENSEARCH_INDEX_DIR="${INDEX_DIR}"
export FRANKENSEARCH_OFFLINE=1
export FRANKENSEARCH_ALLOW_DOWNLOAD=0
export FRANKENSEARCH_LOG="${FRANKENSEARCH_LOG:-error}"
export NO_COLOR=1

CONFIG_DIR="${XDG_CONFIG_HOME}/frankensearch"
CACHE_DIR="${XDG_CACHE_HOME}/frankensearch"
INSTALL_MANIFEST="${CONFIG_DIR}/install-manifest.json"

mkdir -p "${LOG_DIR}" "${INSTALL_BIN_DIR}" "${PROJECT_DIR}/src" "${FRANKENSEARCH_MODEL_DIR}"
mkdir -p "${CONFIG_DIR}" "${CACHE_DIR}"

cleanup() {
  if [[ "${KEEP_TMP}" == "1" ]]; then
    echo "Keeping tmp dir: ${TMP_ROOT}"
    return
  fi
  if [[ -d "${TMP_ROOT}" ]]; then
    rm -r "${TMP_ROOT}"
  fi
}
trap cleanup EXIT

dump_diagnostics() {
  local phase="$1"
  echo "E2E installer smoke failed during phase: ${phase}" >&2
  for file in "${LOG_DIR}"/*.log; do
    [[ -f "${file}" ]] || continue
    echo "--- ${file##*/} ---" >&2
    cat "${file}" >&2
  done
  echo "--- install dir listing ---" >&2
  ls -la "${INSTALL_BIN_DIR}" >&2 || true
  echo "--- index dir listing ---" >&2
  ls -la "${INDEX_DIR}" >&2 || true
}

run_phase() {
  local phase_exit_code="$1"
  local phase_name="$2"
  shift 2

  local log_file="${LOG_DIR}/${phase_name}.log"
  local started
  started="$(date +%s)"

  set +e
  "$@" >"${log_file}" 2>&1
  local status=$?
  set -e

  local elapsed
  elapsed="$(( $(date +%s) - started ))"
  printf 'phase=%s status=%d elapsed_s=%s\n' "${phase_name}" "${status}" "${elapsed}" >>"${LOG_DIR}/timings.log"

  if [[ "${status}" -ne 0 ]]; then
    dump_diagnostics "${phase_name}"
    exit "${phase_exit_code}"
  fi
}

binary_supports_index_dir_flag() {
  local candidate="$1"
  set +e
  "${candidate}" version --index-dir "${TMP_ROOT}/probe-index" >/dev/null 2>&1
  local status=$?
  set -e

  [[ "${status}" -eq 0 ]]
}

install_local_binary() {
  if [[ -n "${FSFS_E2E_FSFS_BIN:-}" && -x "${FSFS_E2E_FSFS_BIN}" ]]; then
    if ! binary_supports_index_dir_flag "${FSFS_E2E_FSFS_BIN}"; then
      echo "Provided FSFS_E2E_FSFS_BIN does not support --index-dir" >&2
      return 1
    fi
    cp "${FSFS_E2E_FSFS_BIN}" "${FSFS_BIN}"
    chmod +x "${FSFS_BIN}"
    return
  fi

  (
    cd "${REPO_ROOT}"
    cargo build -p frankensearch-fsfs --bin fsfs
  )

  for candidate in \
    "${REPO_ROOT}/target/debug/fsfs" \
    "/data/tmp/cargo-target/debug/fsfs"; do
    if [[ -x "${candidate}" ]] && binary_supports_index_dir_flag "${candidate}"; then
      cp "${candidate}" "${FSFS_BIN}"
      chmod +x "${FSFS_BIN}"
      return
    fi
  done

  echo "Unable to locate freshly built fsfs binary" >&2
  return 1
}

install_release_binary() {
  bash "${REPO_ROOT}/scripts/install.sh" \
    --dest "${INSTALL_BIN_DIR}" \
    --force \
    --easy-mode \
    --no-configure \
    --offline
}

phase_install() {
  case "${INSTALL_MODE}" in
    local)
      install_local_binary
      ;;
    release)
      install_release_binary
      ;;
    *)
      echo "Unknown FSFS_E2E_INSTALL_MODE: ${INSTALL_MODE}" >&2
      return 2
      ;;
  esac

  [[ -x "${FSFS_BIN}" ]]
}

phase_prepare_corpus() {
  cat >"${PROJECT_DIR}/src/auth.rs" <<'EOF'
pub fn authenticate_user(token: &str) -> bool {
    token.starts_with("bearer ")
}
EOF

  cat >"${PROJECT_DIR}/src/db.rs" <<'EOF'
pub fn connect_database(url: &str) -> &'static str {
    if url.contains("postgres") { "ok" } else { "invalid" }
}
EOF

  cat >"${PROJECT_DIR}/README.md" <<'EOF'
# Sample Project

This fixture project documents authentication middleware and database connection setup.
EOF

  cat >"${INSTALL_MANIFEST}" <<'EOF'
{"version":"0.1.0-test","install_dir":"~/.local/bin","binary":"fsfs"}
EOF

  : >"${CACHE_DIR}/cache-sentinel.txt"
}

verify_search_contains() {
  local query="$1"
  local expected_path_fragment="$2"
  local log_label="$3"
  local log_file="${LOG_DIR}/${log_label}.log"

  set +e
  "${FSFS_BIN}" search "${query}" \
    --index-dir "${INDEX_DIR}" \
    --no-watch-mode \
    --format json \
    --limit 5 >"${log_file}" 2>&1
  local status=$?
  set -e

  if [[ "${status}" -ne 0 ]]; then
    dump_diagnostics "${log_label}"
    exit 3
  fi

  if command -v jq >/dev/null 2>&1; then
    if ! jq -e --arg needle "${expected_path_fragment}" \
      '.data.hits[:5] | map(.path) | any(contains($needle))' \
      "${log_file}" >/dev/null; then
      dump_diagnostics "${log_label}"
      exit 3
    fi
  else
    if ! grep -q "${expected_path_fragment}" "${log_file}"; then
      dump_diagnostics "${log_label}"
      exit 3
    fi
  fi
}

run_phase 1 install phase_install
run_phase 1 prepare_corpus phase_prepare_corpus

run_phase 2 index "${FSFS_BIN}" index "${PROJECT_DIR}" \
  --index-dir "${INDEX_DIR}" \
  --no-watch-mode \
  --format json

verify_search_contains "authentication" "auth.rs" "search_authentication"
verify_search_contains "database connection" "db.rs" "search_database"

run_phase 3 doctor "${FSFS_BIN}" doctor \
  --index-dir "${INDEX_DIR}" \
  --no-watch-mode \
  --format json

run_phase 4 uninstall "${FSFS_BIN}" uninstall \
  --index-dir "${INDEX_DIR}" \
  --no-watch-mode \
  --yes \
  --purge \
  --format json

for cleanup_target in \
  "${FSFS_BIN}" \
  "${INDEX_DIR}" \
  "${FRANKENSEARCH_MODEL_DIR}" \
  "${CONFIG_DIR}" \
  "${CACHE_DIR}" \
  "${INSTALL_MANIFEST}"; do
  if [[ -e "${cleanup_target}" ]]; then
    dump_diagnostics "verify_cleanup_${cleanup_target##*/}"
    exit 4
  fi
done

echo "E2E SMOKE TEST PASSED"
