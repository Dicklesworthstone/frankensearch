#!/usr/bin/env bash
# rch-ensure-deps.sh — Bootstrap sibling path dependencies for rch workers.
#
# When rch syncs frankensearch to a remote worker via rsync, it only syncs
# the project directory itself. The workspace Cargo.toml references sibling
# path dependencies (asupersync, frankensqlite, fast_cmaes) that don't exist
# on workers. This script clones them as siblings and rewrites absolute paths
# in Cargo.toml files so cargo can resolve all dependencies.
#
# Usage:
#   scripts/rch-ensure-deps.sh              # Auto-detect and fix if needed
#   scripts/rch-ensure-deps.sh --force      # Force re-clone even if present
#   scripts/rch-ensure-deps.sh --check      # Dry-run: report missing deps, exit 1 if any
#
# This script is idempotent and safe to run multiple times.
# It mirrors the CI workflow's "Prepare path dependencies" step.
#
# Context: https://github.com/Dicklesworthstone/frankensearch — bead bd-1pgv

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
# Pin sibling deps to explicit commits for reproducibility.
# These MUST match the refs in .github/workflows/ci.yml.

ASUPERSYNC_REPO="https://github.com/Dicklesworthstone/asupersync.git"
ASUPERSYNC_REF="15e6b6920fa0ad3e6d843ea55186eed754389ad2"

FRANKENSQLITE_REPO="https://github.com/Dicklesworthstone/frankensqlite.git"
FRANKENSQLITE_REF="5c99eeb93d789c1309d5c46a540289369ff39535"

FAST_CMAES_REPO="https://github.com/Dicklesworthstone/fast_cmaes.git"
FAST_CMAES_REF="17f633e2c24bdd0c358310949066e5922b9e17b5"

# ─── Resolve paths ──────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPS_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"

MODE="${1:-auto}"

# ─── Helpers ────────────────────────────────────────────────────────────────

log_info()  { echo "[rch-deps] $*"; }
log_warn()  { echo "[rch-deps] WARNING: $*" >&2; }
log_error() { echo "[rch-deps] ERROR: $*" >&2; }

clone_or_update() {
    local repo_url="$1"
    local dest_path="$2"
    local ref="$3"
    local name
    name="$(basename "${dest_path}")"

    if [[ -d "${dest_path}/.git" ]]; then
        if [[ "${MODE}" == "--force" ]]; then
            log_info "${name}: force-refreshing to ${ref:0:12}..."
            git -C "${dest_path}" fetch --depth 1 origin "${ref}" 2>/dev/null
            git -C "${dest_path}" checkout --detach FETCH_HEAD 2>/dev/null
        else
            log_info "${name}: already present, skipping (use --force to refresh)"
        fi
    else
        log_info "${name}: cloning ${ref:0:12}..."
        git clone --no-checkout "${repo_url}" "${dest_path}" 2>/dev/null
        git -C "${dest_path}" fetch --depth 1 origin "${ref}" 2>/dev/null
        git -C "${dest_path}" checkout --detach FETCH_HEAD 2>/dev/null
    fi
}

check_dep() {
    local dest_path="$1"
    local name
    name="$(basename "${dest_path}")"
    if [[ -d "${dest_path}" ]]; then
        echo "  OK: ${name} (${dest_path})"
        return 0
    else
        echo "  MISSING: ${name} (${dest_path})"
        return 1
    fi
}

needs_path_rewrite() {
    # Check if any Cargo.toml still references /data/projects/ (dev machine paths)
    # that don't resolve on this host.
    if [[ -d "/data/projects/frankensqlite" ]]; then
        return 1  # Paths resolve fine (probably on dev machine)
    fi
    grep -rq '/data/projects/' "${PROJECT_ROOT}"/Cargo.toml \
        "${PROJECT_ROOT}"/crates/*/Cargo.toml \
        "${PROJECT_ROOT}"/tools/*/Cargo.toml 2>/dev/null
}

rewrite_absolute_paths() {
    log_info "Rewriting /data/projects/ paths to ${DEPS_DIR}/..."
    find "${PROJECT_ROOT}" -name Cargo.toml -exec \
        sed -i.rch-bak -e "s|/data/projects/|${DEPS_DIR}/|g" {} +
    find "${PROJECT_ROOT}" -name '*.rch-bak' -delete
}

# ─── Main ───────────────────────────────────────────────────────────────────

if [[ "${MODE}" == "--check" ]]; then
    log_info "Checking sibling dependency availability..."
    missing=0
    check_dep "${DEPS_DIR}/asupersync"    || missing=$((missing + 1))
    check_dep "${DEPS_DIR}/frankensqlite"  || missing=$((missing + 1))
    check_dep "${DEPS_DIR}/fast_cmaes"     || missing=$((missing + 1))

    if needs_path_rewrite; then
        echo "  NOTE: Cargo.toml files contain /data/projects/ paths that need rewriting"
        missing=$((missing + 1))
    fi

    if [[ "${missing}" -gt 0 ]]; then
        log_warn "${missing} issue(s) found. Run without --check to fix."
        exit 1
    else
        log_info "All dependencies available."
        exit 0
    fi
fi

# Auto mode: skip if deps already exist (unless --force)
if [[ "${MODE}" == "auto" ]]; then
    all_present=true
    [[ -d "${DEPS_DIR}/asupersync" ]]    || all_present=false
    [[ -d "${DEPS_DIR}/frankensqlite" ]]  || all_present=false
    [[ -d "${DEPS_DIR}/fast_cmaes" ]]     || all_present=false

    if ${all_present} && ! needs_path_rewrite; then
        log_info "All sibling deps present and paths resolve. Nothing to do."
        exit 0
    fi
fi

# Clone/update sibling dependencies
log_info "Ensuring sibling dependencies in ${DEPS_DIR}/..."
clone_or_update "${ASUPERSYNC_REPO}"    "${DEPS_DIR}/asupersync"    "${ASUPERSYNC_REF}"
clone_or_update "${FRANKENSQLITE_REPO}" "${DEPS_DIR}/frankensqlite" "${FRANKENSQLITE_REF}"
clone_or_update "${FAST_CMAES_REPO}"    "${DEPS_DIR}/fast_cmaes"    "${FAST_CMAES_REF}"

# Rewrite absolute paths if needed (worker doesn't have /data/projects/)
if needs_path_rewrite; then
    rewrite_absolute_paths
fi

log_info "Done. Sibling dependencies ready."
