#!/usr/bin/env bash
# rch-ensure-deps.sh — Bootstrap sibling path dependencies for rch workers.
#
# When rch syncs frankensearch to a remote worker via rsync, it only syncs
# the project directory itself. The workspace Cargo.toml references sibling
# path dependencies (asupersync, frankensqlite, fast_cmaes, frankentui)
# that don't exist on workers by default.
#
# Local usage:
#   scripts/rch-ensure-deps.sh              # Auto-detect and fix if needed
#   scripts/rch-ensure-deps.sh --force      # Force re-clone even if present
#   scripts/rch-ensure-deps.sh --check      # Dry-run: report missing deps, exit 1 if any
#
# Worker usage:
#   scripts/rch-ensure-deps.sh --all-workers
#   scripts/rch-ensure-deps.sh --all-workers --check
#   scripts/rch-ensure-deps.sh --worker vmi1152480 --force
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

FRANKENTUI_REPO="https://github.com/Dicklesworthstone/frankentui.git"
FRANKENTUI_REF="abbb39cd88277de40e88734e40a41a575935a863"

RCH_REMOTE_DEPS_DIR="${RCH_REMOTE_DEPS_DIR:-/tmp/rch/frankensearch}"

# ─── Resolve paths ──────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPS_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"

# ─── Args ───────────────────────────────────────────────────────────────────

MODE="auto"
TARGET_WORKER=""
ALL_WORKERS=false

usage() {
    cat <<'EOF'
Usage:
  scripts/rch-ensure-deps.sh [MODE] [--all-workers | --worker <worker-id-or-host>]

Modes:
  auto      (default) clone missing deps only
  --check   report missing deps and exit 1 when incomplete
  --force   refresh existing deps to pinned refs

Examples:
  scripts/rch-ensure-deps.sh
  scripts/rch-ensure-deps.sh --check
  scripts/rch-ensure-deps.sh --all-workers
  scripts/rch-ensure-deps.sh --all-workers --check
  scripts/rch-ensure-deps.sh --worker vmi1152480 --force
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        auto|--check|--force)
            MODE="$1"
            shift
            ;;
        --all-workers)
            ALL_WORKERS=true
            shift
            ;;
        --worker)
            if [[ $# -lt 2 ]]; then
                echo "[rch-deps] ERROR: --worker requires <worker-id-or-host>" >&2
                exit 2
            fi
            TARGET_WORKER="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[rch-deps] ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${ALL_WORKERS}" == true && -n "${TARGET_WORKER}" ]]; then
    echo "[rch-deps] ERROR: use either --all-workers or --worker, not both" >&2
    exit 2
fi

# ─── Helpers ────────────────────────────────────────────────────────────────

log_info()  { echo "[rch-deps] $*"; }
log_warn()  { echo "[rch-deps] WARNING: $*" >&2; }
log_error() { echo "[rch-deps] ERROR: $*" >&2; }

clone_or_update() {
    local repo_url="$1"
    local dest_path="$2"
    local ref="$3"
    local mode="$4"
    local name
    name="$(basename "${dest_path}")"

    if [[ -d "${dest_path}/.git" ]]; then
        if [[ "${mode}" == "--force" ]]; then
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

run_local_bootstrap() {
    local mode="$1"

    if [[ "${mode}" == "--check" ]]; then
        log_info "Checking sibling dependency availability..."
        local missing=0
        check_dep "${DEPS_DIR}/asupersync"    || missing=$((missing + 1))
        check_dep "${DEPS_DIR}/frankensqlite" || missing=$((missing + 1))
        check_dep "${DEPS_DIR}/fast_cmaes"    || missing=$((missing + 1))
        check_dep "${DEPS_DIR}/frankentui"    || missing=$((missing + 1))

        if needs_path_rewrite; then
            echo "  NOTE: Cargo.toml files contain /data/projects/ paths that need rewriting"
            missing=$((missing + 1))
        fi

        if [[ "${missing}" -gt 0 ]]; then
            log_warn "${missing} issue(s) found. Run without --check to fix."
            return 1
        fi
        log_info "All dependencies available."
        return 0
    fi

    if [[ "${mode}" == "auto" ]]; then
        local all_present=true
        [[ -d "${DEPS_DIR}/asupersync" ]]    || all_present=false
        [[ -d "${DEPS_DIR}/frankensqlite" ]] || all_present=false
        [[ -d "${DEPS_DIR}/fast_cmaes" ]]    || all_present=false
        [[ -d "${DEPS_DIR}/frankentui" ]]    || all_present=false

        if ${all_present} && ! needs_path_rewrite; then
            log_info "All sibling deps present and paths resolve. Nothing to do."
            return 0
        fi
    fi

    log_info "Ensuring sibling dependencies in ${DEPS_DIR}/..."
    clone_or_update "${ASUPERSYNC_REPO}"    "${DEPS_DIR}/asupersync"    "${ASUPERSYNC_REF}" "${mode}"
    clone_or_update "${FRANKENSQLITE_REPO}" "${DEPS_DIR}/frankensqlite" "${FRANKENSQLITE_REF}" "${mode}"
    clone_or_update "${FAST_CMAES_REPO}"    "${DEPS_DIR}/fast_cmaes"    "${FAST_CMAES_REF}" "${mode}"
    clone_or_update "${FRANKENTUI_REPO}"    "${DEPS_DIR}/frankentui"    "${FRANKENTUI_REF}" "${mode}"

    if needs_path_rewrite; then
        rewrite_absolute_paths
    fi

    log_info "Done. Sibling dependencies ready."
}

list_workers_from_rch() {
    if ! command -v rch >/dev/null 2>&1; then
        log_error "rch command is required for --all-workers"
        return 1
    fi

    local workers_json
    if ! workers_json="$(rch workers list --json 2>/dev/null)"; then
        log_error "failed to query workers via 'rch workers list --json'"
        return 1
    fi

    awk -F'"' '/"id"[[:space:]]*:/ { print $4 }' <<<"${workers_json}"
}

bootstrap_remote_worker() {
    local worker="$1"

    log_info "Bootstrapping ${worker}:${RCH_REMOTE_DEPS_DIR} (${MODE})"

    ssh -o BatchMode=yes -o ConnectTimeout=10 "${worker}" \
        bash -s -- "${MODE}" "${RCH_REMOTE_DEPS_DIR}" \
        "${ASUPERSYNC_REPO}" "${ASUPERSYNC_REF}" \
        "${FRANKENSQLITE_REPO}" "${FRANKENSQLITE_REF}" \
        "${FAST_CMAES_REPO}" "${FAST_CMAES_REF}" \
        "${FRANKENTUI_REPO}" "${FRANKENTUI_REF}" <<'EOF'
set -euo pipefail

mode="$1"
deps_dir="$2"
asupersync_repo="$3"
asupersync_ref="$4"
frankensqlite_repo="$5"
frankensqlite_ref="$6"
fast_cmaes_repo="$7"
fast_cmaes_ref="$8"
frankentui_repo="${9:-}"
frankentui_ref="${10:-}"

log()  { echo "[rch-deps][remote] $*"; }
warn() { echo "[rch-deps][remote] WARNING: $*" >&2; }

clone_or_update_remote() {
    local repo_url="$1"
    local dest_path="$2"
    local ref="$3"
    local name
    name="$(basename "${dest_path}")"

    if [[ -d "${dest_path}/.git" ]]; then
        if [[ "${mode}" == "--force" ]]; then
            log "${name}: force-refreshing to ${ref:0:12}..."
            git -C "${dest_path}" fetch --depth 1 origin "${ref}" 2>/dev/null
            git -C "${dest_path}" checkout --detach FETCH_HEAD 2>/dev/null
        else
            log "${name}: already present, skipping (use --force to refresh)"
        fi
    else
        log "${name}: cloning ${ref:0:12}..."
        git clone --no-checkout "${repo_url}" "${dest_path}" 2>/dev/null
        git -C "${dest_path}" fetch --depth 1 origin "${ref}" 2>/dev/null
        git -C "${dest_path}" checkout --detach FETCH_HEAD 2>/dev/null
    fi
}

check_dep_remote() {
    local path="$1"
    local name
    name="$(basename "${path}")"
    if [[ -d "${path}" ]]; then
        echo "  OK: ${name} (${path})"
        return 0
    else
        echo "  MISSING: ${name} (${path})"
        return 1
    fi
}

mkdir -p "${deps_dir}"

if [[ "${mode}" == "--check" ]]; then
    log "Checking remote dependency availability in ${deps_dir}..."
    missing=0
    check_dep_remote "${deps_dir}/asupersync" || missing=$((missing + 1))
    check_dep_remote "${deps_dir}/frankensqlite" || missing=$((missing + 1))
    check_dep_remote "${deps_dir}/fast_cmaes" || missing=$((missing + 1))
    check_dep_remote "${deps_dir}/frankentui" || missing=$((missing + 1))
    if [[ "${missing}" -gt 0 ]]; then
        warn "${missing} issue(s) found"
        exit 1
    fi
    log "Remote dependencies available."
    exit 0
fi

log "Ensuring remote sibling dependencies in ${deps_dir}..."
clone_or_update_remote "${asupersync_repo}" "${deps_dir}/asupersync" "${asupersync_ref}"
clone_or_update_remote "${frankensqlite_repo}" "${deps_dir}/frankensqlite" "${frankensqlite_ref}"
clone_or_update_remote "${fast_cmaes_repo}" "${deps_dir}/fast_cmaes" "${fast_cmaes_ref}"
if [[ -n "${frankentui_repo}" && -n "${frankentui_ref}" ]]; then
    clone_or_update_remote "${frankentui_repo}" "${deps_dir}/frankentui" "${frankentui_ref}"
fi
log "Done."
EOF
}

run_remote_bootstrap() {
    local -a workers=()

    if [[ -n "${TARGET_WORKER}" ]]; then
        workers=("${TARGET_WORKER}")
    elif [[ "${ALL_WORKERS}" == true ]]; then
        mapfile -t workers < <(list_workers_from_rch)
        if [[ ${#workers[@]} -eq 0 ]]; then
            log_error "no workers found from 'rch workers list --json'"
            return 1
        fi
    else
        return 1
    fi

    local failures=0
    local worker
    for worker in "${workers[@]}"; do
        if ! bootstrap_remote_worker "${worker}"; then
            log_error "bootstrap failed for ${worker}"
            failures=$((failures + 1))
        fi
    done

    if [[ "${failures}" -gt 0 ]]; then
        log_error "remote bootstrap failed on ${failures} worker(s)"
        return 1
    fi

    log_info "Remote bootstrap complete for ${#workers[@]} worker(s)."
    return 0
}

# ─── Main ───────────────────────────────────────────────────────────────────

if [[ -n "${TARGET_WORKER}" || "${ALL_WORKERS}" == true ]]; then
    run_remote_bootstrap
else
    run_local_bootstrap "${MODE}"
fi
