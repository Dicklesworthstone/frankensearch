#!/usr/bin/env bash
# rch-ensure-deps.sh — Bootstrap sibling path dependencies for rch workers.
#
# When rch syncs frankensearch to a remote worker via rsync, it only syncs
# the project directory itself. The workspace Cargo.toml references sibling
# path dependencies (asupersync, frankensqlite, fast_cmaes, frankentui)
# that don't exist on workers by default.
#
# In worker mode this also warms cargo's SHARED caches (crates.io index + the
# frankentorch git dependency). Those are not part of the synced project, so a
# cold worker-scoped target pool fetches them over the network *before* it can
# start compiling — and that has repeatedly pushed benchmark jobs past their
# admission window without ever linking a binary (docs/NEGATIVE_EVIDENCE.md:
# bd-l5x3, bd-3srq, bd-r3rd). `--check` now reports a cold cargo git cache as
# an issue so the condition is visible before a measurement window is spent.
#
# Local usage:
#   scripts/rch-ensure-deps.sh              # Auto-detect and fix if needed
#   scripts/rch-ensure-deps.sh --force      # Force re-clone even if present
#   scripts/rch-ensure-deps.sh --check      # Dry-run: report missing deps, exit 1 if any
#   scripts/rch-ensure-deps.sh --models     # Provision pinned bundled-model inputs
#   scripts/rch-ensure-deps.sh --models-only # Provision/check only bundled-model inputs
#
# Worker usage:
#   scripts/rch-ensure-deps.sh --all-workers --models-only
#   scripts/rch-ensure-deps.sh --all-workers --models-only --check
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
FAST_CMAES_REF="9406d5ec9512767106c9639628e30902ef7eae32"

FRANKENTUI_REPO="https://github.com/Dicklesworthstone/frankentui.git"
FRANKENTUI_REF="4f2803a7c99d4fc439f3503e93c69e9ca68f354c"

# Match RCH's configured remote_base (`/data/tmp/rch`). Some workers install
# the `rch` executable at `/tmp/rch`, so that legacy path cannot be a directory.
RCH_REMOTE_DEPS_DIR="${RCH_REMOTE_DEPS_DIR:-/data/tmp/rch/frankensearch}"

# Optional explicit worker-side project path used to warm Cargo's shared caches.
# RCH 1.0.52 normally syncs into a content-addressed directory below
# `${RCH_REMOTE_DEPS_DIR}` (for example
# `/data/tmp/rch/frankensearch/<project-hash>`), so an unset override is
# discovered remotely after at least one sync.
RCH_REMOTE_PROJECT_DIR="${RCH_REMOTE_PROJECT_DIR:-}"

# Workspace git dependency (frankentorch: ft-api / ft-autograd / ft-core), pinned
# in the root Cargo.toml. Unlike the sibling PATH deps above, cargo resolves this
# over the network, and it is fetched during workspace resolution even for a
# build that does not link it. On a cold worker-scoped target pool that fetch —
# together with the crates.io index update — runs before compilation starts and
# has repeatedly pushed benchmark jobs past their admission window without ever
# linking (see docs/NEGATIVE_EVIDENCE.md: bd-l5x3, bd-3srq, bd-r3rd).
FRANKENTORCH_REF="c305306b251753099620ad5fe02e78c07c167cf6"

# `frankensearch-embed` keeps build.rs network-free. Default-feature fsfs
# builds therefore require these exact manifest inputs to exist before Cargo
# starts. Keep this list byte-for-byte aligned with
# crates/frankensearch-embed/build.rs and model_manifest.rs.
BUNDLED_MODEL_SPECS=(
    "potion-multilingual-128M|tokenizer.json|https://huggingface.co/minishlab/potion-multilingual-128M/resolve/a28f4eebecd4dc585034f605e52d414878a0417c/tokenizer.json|19f1909063da3cfe3bd83a782381f040dccea475f4816de11116444a73e1b6a1|18616131"
    "potion-multilingual-128M|model.safetensors|https://huggingface.co/minishlab/potion-multilingual-128M/resolve/a28f4eebecd4dc585034f605e52d414878a0417c/model.safetensors|14b5eb39cb4ce5666da8ad1f3dc6be4346e9b2d601c073302fa0a31bf7943397|512361560"
    "all-MiniLM-L6-v2|onnx/model.onnx|https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model.onnx|6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452|90405214"
    "all-MiniLM-L6-v2|tokenizer.json|https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json|be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037|466247"
    "all-MiniLM-L6-v2|config.json|https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json|953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41|612"
    "all-MiniLM-L6-v2|special_tokens_map.json|https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/special_tokens_map.json|303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3|112"
    "all-MiniLM-L6-v2|tokenizer_config.json|https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json|acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b|350"
)

# ─── Resolve paths ──────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPS_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"

# ─── Args ───────────────────────────────────────────────────────────────────

MODE="auto"
TARGET_WORKER=""
ALL_WORKERS=false
PROVISION_MODELS=false
MODELS_ONLY=false

usage() {
    cat <<'EOF'
Usage:
  scripts/rch-ensure-deps.sh [MODE] [--models | --models-only] [--all-workers | --worker <worker-id-or-host>]

Modes:
  auto      (default) clone missing deps only
  --check   report missing deps and exit 1 when incomplete
  --force   refresh existing deps to pinned refs
  --models  also provision the exact SHA-256-pinned bundled-model inputs
  --models-only
             provision/check bundled-model inputs without sibling-dependency work

Examples:
  scripts/rch-ensure-deps.sh
  scripts/rch-ensure-deps.sh --check
  scripts/rch-ensure-deps.sh --models
  scripts/rch-ensure-deps.sh --all-workers --models-only
  scripts/rch-ensure-deps.sh --all-workers --models-only --check
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
        --models)
            PROVISION_MODELS=true
            shift
            ;;
        --models-only)
            PROVISION_MODELS=true
            MODELS_ONLY=true
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

validate_model_specs() {
    local spec model relative_path url sha256 size destination
    local -A seen_destinations=()
    for spec in "${BUNDLED_MODEL_SPECS[@]}"; do
        IFS='|' read -r model relative_path url sha256 size <<<"${spec}"
        if [[ -z "${model}" || "${model}" == */* || -z "${relative_path}" || "${relative_path}" == /* ]] ||
            [[ "/${relative_path}/" == *"/../"* || "/${relative_path}/" == *"/./"* ]]; then
            log_error "invalid bundled-model destination in spec: ${spec}"
            return 1
        fi
        if [[ "${url}" != https://huggingface.co/* ]]; then
            log_error "bundled-model URL is not pinned to Hugging Face HTTPS: ${url}"
            return 1
        fi
        if [[ ! "${sha256}" =~ ^[[:xdigit:]]{64}$ ]]; then
            log_error "invalid bundled-model SHA-256 for ${model}/${relative_path}"
            return 1
        fi
        if [[ ! "${size}" =~ ^[1-9][0-9]*$ ]]; then
            log_error "invalid bundled-model byte size for ${model}/${relative_path}"
            return 1
        fi
        destination="${model}/${relative_path}"
        if [[ -n "${seen_destinations[${destination}]:-}" ]]; then
            log_error "duplicate bundled-model destination: ${destination}"
            return 1
        fi
        seen_destinations["${destination}"]=1
    done
}

file_sha256() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{ print $1 }'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${path}" | awk '{ print $1 }'
    else
        log_error "sha256sum or shasum is required to verify bundled models"
        return 1
    fi
}

artifact_matches() {
    local path="$1"
    local expected_sha256="$2"
    local expected_size="$3"
    [[ -f "${path}" ]] || return 1
    [[ "$(wc -c <"${path}")" -eq "${expected_size}" ]] || return 1
    [[ "$(file_sha256 "${path}")" == "${expected_sha256}" ]]
}

check_bundled_models() {
    local model_root="$1"
    local missing=0
    local spec model relative_path url sha256 size destination
    for spec in "${BUNDLED_MODEL_SPECS[@]}"; do
        IFS='|' read -r model relative_path url sha256 size <<<"${spec}"
        destination="${model_root}/${model}/${relative_path}"
        if artifact_matches "${destination}" "${sha256}" "${size}"; then
            echo "  OK: bundled model ${model}/${relative_path}"
        else
            echo "  MISSING/INVALID: bundled model ${model}/${relative_path}"
            missing=$((missing + 1))
        fi
    done
    [[ "${missing}" -eq 0 ]]
}

provision_bundled_models() {
    local model_root="$1"
    local spec model relative_path url sha256 size destination partial
    command -v curl >/dev/null 2>&1 || {
        log_error "curl is required to provision bundled models"
        return 1
    }
    for spec in "${BUNDLED_MODEL_SPECS[@]}"; do
        IFS='|' read -r model relative_path url sha256 size <<<"${spec}"
        destination="${model_root}/${model}/${relative_path}"
        if artifact_matches "${destination}" "${sha256}" "${size}"; then
            log_info "bundled model ${model}/${relative_path}: verified"
            continue
        fi
        if [[ -e "${destination}" ]]; then
            log_error "refusing to overwrite invalid bundled artifact ${destination}"
            return 1
        fi
        mkdir -p "$(dirname "${destination}")"
        partial="${destination}.partial"
        if [[ -e "${partial}" ]] && [[ ! -f "${partial}" ]]; then
            log_error "partial artifact is not a regular file: ${partial}"
            return 1
        fi
        if ! artifact_matches "${partial}" "${sha256}" "${size}"; then
            log_info "downloading ${model}/${relative_path}..."
            curl --fail --location --silent --show-error \
                --retry 8 --retry-delay 1 --retry-all-errors \
                --continue-at - --output "${partial}" "${url}"
        fi
        if ! artifact_matches "${partial}" "${sha256}" "${size}"; then
            log_error "downloaded artifact failed size/SHA-256 verification: ${partial}"
            return 1
        fi
        if [[ -e "${destination}" ]]; then
            log_error "destination appeared during download; refusing overwrite: ${destination}"
            return 1
        fi
        mv -n -- "${partial}" "${destination}"
        if ! artifact_matches "${destination}" "${sha256}" "${size}"; then
            log_error "installed artifact failed verification: ${destination}"
            return 1
        fi
        log_info "bundled model ${model}/${relative_path}: installed and verified"
    done
}

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
    elif [[ -e "${dest_path}" ]]; then
        log_error "refusing to initialize non-Git path ${dest_path}"
        return 1
    else
        log_info "${name}: fetching pinned commit ${ref:0:12}..."
        mkdir -p "${dest_path}"
        git -C "${dest_path}" init --quiet
        git -C "${dest_path}" remote add origin "${repo_url}"
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
    local model_root
    model_root="${FRANKENSEARCH_BUNDLED_MODELS_SOURCE_DIR:-${FRANKENSEARCH_MODEL_DIR:-${HOME}/.local/share/frankensearch/models}}"

    if [[ "${MODELS_ONLY}" == true ]]; then
        if [[ "${mode}" == "--check" ]]; then
            log_info "Checking bundled-model inputs..."
            check_bundled_models "${model_root}"
        else
            provision_bundled_models "${model_root}"
        fi
        return
    fi

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
        if [[ "${PROVISION_MODELS}" == true ]] && ! check_bundled_models "${model_root}"; then
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
            if [[ "${PROVISION_MODELS}" == true ]]; then
                provision_bundled_models "${model_root}"
            fi
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
    if [[ "${PROVISION_MODELS}" == true ]]; then
        provision_bundled_models "${model_root}"
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

resolve_worker_target() {
    local requested="$1"
    if [[ "${requested}" == *@* ]]; then
        printf '%s\n' "${requested}"
        return 0
    fi
    if ! command -v rch >/dev/null 2>&1; then
        printf '%s\n' "${requested}"
        return 0
    fi

    local workers_json resolved
    workers_json="$(rch workers list --json 2>/dev/null)" || {
        printf '%s\n' "${requested}"
        return 0
    }
    resolved="$(
        awk -F'"' -v requested="${requested}" '
            /"id"[[:space:]]*:/ { id = $4 }
            /"host"[[:space:]]*:/ { host = $4 }
            /"user"[[:space:]]*:/ {
                user = $4
                if (id == requested || host == requested) {
                    print user "@" host
                    exit
                }
                id = ""
                host = ""
                user = ""
            }
        ' <<<"${workers_json}"
    )"
    printf '%s\n' "${resolved:-${requested}}"
}

resolve_worker_identity_file() {
    local requested="$1"
    local lookup="${requested#*@}"
    local config_file="${RCH_WORKERS_CONFIG_FILE:-${HOME}/.config/rch/workers.toml}"
    [[ -f "${config_file}" ]] || return 0

    local identity_file
    identity_file="$(
        awk -F'"' -v requested="${lookup}" '
            /^\[\[workers\]\]/ {
                if ((id == requested || host == requested) && identity != "") {
                    print identity
                    found = 1
                    exit
                }
                id = ""
                host = ""
                identity = ""
            }
            /^id[[:space:]]*=/ { id = $2 }
            /^host[[:space:]]*=/ { host = $2 }
            /^identity_file[[:space:]]*=/ { identity = $2 }
            END {
                if (!found && (id == requested || host == requested) && identity != "") {
                    print identity
                }
            }
        ' "${config_file}"
    )"
    if [[ "${identity_file}" == "~/"* ]]; then
        identity_file="${HOME}/${identity_file:2}"
    fi
    printf '%s\n' "${identity_file}"
}

bootstrap_remote_worker() {
    local requested_worker="$1"
    local worker identity_file remote_project_dir_arg
    worker="$(resolve_worker_target "${requested_worker}")"
    identity_file="$(resolve_worker_identity_file "${requested_worker}")"
    # OpenSSH command serialization does not preserve an empty positional
    # argument. Use a non-empty sentinel so later flags cannot shift left when
    # the worker should auto-discover its content-addressed project root.
    remote_project_dir_arg="${RCH_REMOTE_PROJECT_DIR:-__RCH_DISCOVER_PROJECT__}"
    local -a encoded_model_specs=()
    local -a ssh_options=(-o BatchMode=yes -o ConnectTimeout=10)
    local spec

    if [[ -n "${identity_file}" ]]; then
        ssh_options+=(-o IdentitiesOnly=yes -i "${identity_file}")
    fi
    if [[ "${PROVISION_MODELS}" == true ]]; then
        command -v base64 >/dev/null 2>&1 || {
            log_error "base64 is required to transport bundled-model specs"
            return 1
        }
        for spec in "${BUNDLED_MODEL_SPECS[@]}"; do
            encoded_model_specs+=("$(printf '%s' "${spec}" | base64 | tr -d '\n')")
        done
    fi

    log_info "Bootstrapping ${worker}:${RCH_REMOTE_DEPS_DIR} (${MODE})"

    ssh "${ssh_options[@]}" "${worker}" \
        bash -s -- "${MODE}" "${RCH_REMOTE_DEPS_DIR}" \
        "${ASUPERSYNC_REPO}" "${ASUPERSYNC_REF}" \
        "${FRANKENSQLITE_REPO}" "${FRANKENSQLITE_REF}" \
        "${FAST_CMAES_REPO}" "${FAST_CMAES_REF}" \
        "${FRANKENTUI_REPO}" "${FRANKENTUI_REF}" \
        "${remote_project_dir_arg}" "${FRANKENTORCH_REF}" \
        "${PROVISION_MODELS}" "${MODELS_ONLY}" "${encoded_model_specs[@]}" <<'EOF'
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
project_dir="${11:-}"
frankentorch_ref="${12:-}"
provision_models="${13:-false}"
models_only="${14:-false}"
shift 14
encoded_model_specs=("$@")
if [[ "${project_dir}" == "__RCH_DISCOVER_PROJECT__" ]]; then
    project_dir=""
fi
model_specs=()
model_root="${FRANKENSEARCH_BUNDLED_MODELS_SOURCE_DIR:-${FRANKENSEARCH_MODEL_DIR:-${HOME}/.local/share/frankensearch/models}}"

log()  { echo "[rch-deps][remote] $*"; }
warn() { echo "[rch-deps][remote] WARNING: $*" >&2; }

if [[ "${provision_models}" == true ]]; then
    command -v base64 >/dev/null 2>&1 || {
        warn "base64 is required to decode bundled-model specs"
        exit 1
    }
    for encoded_spec in "${encoded_model_specs[@]}"; do
        model_specs+=("$(printf '%s' "${encoded_spec}" | base64 --decode)")
    done
fi

file_sha256_remote() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{ print $1 }'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${path}" | awk '{ print $1 }'
    else
        warn "sha256sum or shasum is required to verify bundled models"
        return 1
    fi
}

artifact_matches_remote() {
    local path="$1"
    local expected_sha256="$2"
    local expected_size="$3"
    [[ -f "${path}" ]] || return 1
    [[ "$(wc -c <"${path}")" -eq "${expected_size}" ]] || return 1
    [[ "$(file_sha256_remote "${path}")" == "${expected_sha256}" ]]
}

check_bundled_models_remote() {
    local missing=0
    local spec model relative_path url sha256 size destination
    for spec in "${model_specs[@]}"; do
        IFS='|' read -r model relative_path url sha256 size <<<"${spec}"
        destination="${model_root}/${model}/${relative_path}"
        if artifact_matches_remote "${destination}" "${sha256}" "${size}"; then
            echo "  OK: bundled model ${model}/${relative_path}"
        else
            echo "  MISSING/INVALID: bundled model ${model}/${relative_path}"
            missing=$((missing + 1))
        fi
    done
    [[ "${missing}" -eq 0 ]]
}

provision_bundled_models_remote() {
    local spec model relative_path url sha256 size destination partial
    command -v curl >/dev/null 2>&1 || {
        warn "curl is required to provision bundled models"
        return 1
    }
    for spec in "${model_specs[@]}"; do
        IFS='|' read -r model relative_path url sha256 size <<<"${spec}"
        destination="${model_root}/${model}/${relative_path}"
        if artifact_matches_remote "${destination}" "${sha256}" "${size}"; then
            log "bundled model ${model}/${relative_path}: verified"
            continue
        fi
        if [[ -e "${destination}" ]]; then
            warn "refusing to overwrite invalid bundled artifact ${destination}"
            return 1
        fi
        mkdir -p "$(dirname "${destination}")"
        partial="${destination}.partial"
        if [[ -e "${partial}" ]] && [[ ! -f "${partial}" ]]; then
            warn "partial artifact is not a regular file: ${partial}"
            return 1
        fi
        if ! artifact_matches_remote "${partial}" "${sha256}" "${size}"; then
            log "downloading ${model}/${relative_path}..."
            curl --fail --location --silent --show-error \
                --retry 8 --retry-delay 1 --retry-all-errors \
                --continue-at - --output "${partial}" "${url}"
        fi
        if ! artifact_matches_remote "${partial}" "${sha256}" "${size}"; then
            warn "downloaded artifact failed size/SHA-256 verification: ${partial}"
            return 1
        fi
        if [[ -e "${destination}" ]]; then
            warn "destination appeared during download; refusing overwrite: ${destination}"
            return 1
        fi
        mv -n -- "${partial}" "${destination}"
        artifact_matches_remote "${destination}" "${sha256}" "${size}" || {
            warn "installed artifact failed verification: ${destination}"
            return 1
        }
        log "bundled model ${model}/${relative_path}: installed and verified"
    done
}

clone_or_update_remote() {
    local repo_url="$1"
    local dest_path="$2"
    local ref="$3"
    local name current_ref
    name="$(basename "${dest_path}")"

    if [[ -d "${dest_path}/.git" ]]; then
        current_ref="$(git -C "${dest_path}" rev-parse HEAD 2>/dev/null || true)"
        if [[ "${current_ref}" == "${ref}" ]]; then
            log "${name}: pinned commit ${ref:0:12} verified"
        elif [[ "${mode}" == "--force" ]]; then
            log "${name}: force-refreshing to ${ref:0:12}..."
            git -C "${dest_path}" fetch --depth 1 origin "${ref}" 2>/dev/null
            git -C "${dest_path}" checkout --detach FETCH_HEAD 2>/dev/null
        else
            warn "${name}: HEAD ${current_ref:-unknown} does not match pinned ${ref}; use --force"
            return 1
        fi
    elif [[ -e "${dest_path}" ]]; then
        warn "refusing to initialize non-Git path ${dest_path}"
        return 1
    else
        log "${name}: fetching pinned commit ${ref:0:12}..."
        mkdir -p "${dest_path}"
        git -C "${dest_path}" init --quiet
        git -C "${dest_path}" remote add origin "${repo_url}"
        git -C "${dest_path}" fetch --depth 1 origin "${ref}" 2>/dev/null
        git -C "${dest_path}" checkout --detach FETCH_HEAD 2>/dev/null
    fi
    current_ref="$(git -C "${dest_path}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${current_ref}" != "${ref}" ]]; then
        warn "${name}: failed to establish pinned commit ${ref}"
        return 1
    fi
}

check_dep_remote() {
    local path="$1"
    local expected_ref="$2"
    local name
    name="$(basename "${path}")"
    local current_ref
    current_ref="$(git -C "${path}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${current_ref}" == "${expected_ref}" ]]; then
        echo "  OK: ${name} (${path}, ${current_ref:0:12})"
        return 0
    elif [[ -d "${path}" ]]; then
        echo "  MISMATCH: ${name} (${path}, HEAD ${current_ref:-unknown}, expected ${expected_ref})"
        return 1
    else
        echo "  MISSING: ${name} (${path})"
        return 1
    fi
}

# Is the pinned frankentorch revision already in cargo's git database? If not,
# the next cold build pays a network fetch before it can start compiling.
cargo_git_cache_warm() {
    local cargo_home="${CARGO_HOME:-${HOME}/.cargo}"
    local db_dir="${cargo_home}/git/db"
    [[ -d "${db_dir}" ]] || return 1
    local repo
    for repo in "${db_dir}"/frankentorch-*; do
        [[ -d "${repo}" ]] || continue
        if git --git-dir="${repo}" cat-file -e "${frankentorch_ref}^{commit}" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

if [[ "${models_only}" == true ]]; then
    if [[ "${mode}" == "--check" ]]; then
        log "Checking bundled-model inputs..."
        check_bundled_models_remote
    else
        provision_bundled_models_remote
    fi
    exit
fi

# Populate the crates.io index and every git dependency in cargo's shared caches
# so a cold worker-scoped target pool starts compiling immediately instead of
# spending its admission window on the network.
resolve_remote_project_manifest() {
    local candidate candidate_dir candidate_name newest=""

    if [[ -n "${project_dir}" && -f "${project_dir}/Cargo.toml" ]]; then
        printf '%s\n' "${project_dir}/Cargo.toml"
        return 0
    fi

    # RCH's remote path is `${remote_base}/${project_id}/${project_hash}`.
    # The sibling-dependency bootstrap lives at the project-id level, so the
    # content-addressed project roots are its immediate children. Exclude the
    # fixed sibling clones and require a frankensearch-specific workspace
    # member before accepting a candidate.
    for candidate in "${deps_dir}"/*/Cargo.toml; do
        [[ -f "${candidate}" ]] || continue
        candidate_dir="$(dirname "${candidate}")"
        candidate_name="$(basename "${candidate_dir}")"
        case "${candidate_name}" in
            asupersync|fast_cmaes|frankensqlite|frankentui)
                continue
                ;;
        esac
        grep -Fq '"crates/frankensearch-core"' "${candidate}" || continue
        if [[ -z "${newest}" || "${candidate}" -nt "${newest}" ]]; then
            newest="${candidate}"
        fi
    done

    [[ -n "${newest}" ]] || return 1
    printf '%s\n' "${newest}"
}

warm_cargo_caches_remote() {
    local manifest
    if ! manifest="$(resolve_remote_project_manifest)"; then
        log "cargo caches: no synced frankensearch manifest below ${deps_dir}; skipping warm"
        return 0
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        warn "cargo caches: cargo not on PATH, skipping warm"
        return 0
    fi
    if [[ "${mode}" != "--force" ]] && cargo_git_cache_warm; then
        log "cargo caches: frankentorch ${frankentorch_ref:0:12} already cached, skipping"
        return 0
    fi
    log "cargo caches: fetching registry index + git deps from ${manifest} (one-time)..."
    # Never fail the bootstrap on a fetch problem — a cold cache is slow, not broken.
    if cargo fetch --locked --manifest-path "${manifest}" >/dev/null 2>&1; then
        log "cargo caches: warm."
    else
        warn "cargo caches: 'cargo fetch --locked' failed; cold builds stay slow"
    fi
}

mkdir -p "${deps_dir}"

if [[ "${mode}" == "--check" ]]; then
    log "Checking remote dependency availability in ${deps_dir}..."
    missing=0
    check_dep_remote "${deps_dir}/asupersync" "${asupersync_ref}" || missing=$((missing + 1))
    check_dep_remote "${deps_dir}/frankensqlite" "${frankensqlite_ref}" || missing=$((missing + 1))
    check_dep_remote "${deps_dir}/fast_cmaes" "${fast_cmaes_ref}" || missing=$((missing + 1))
    check_dep_remote "${deps_dir}/frankentui" "${frankentui_ref}" || missing=$((missing + 1))
    if cargo_git_cache_warm; then
        echo "  OK: cargo git cache (frankentorch ${frankentorch_ref:0:12})"
    else
        echo "  COLD: cargo git cache (frankentorch ${frankentorch_ref:0:12}) — next cold build pays a network fetch"
        missing=$((missing + 1))
    fi
    if [[ "${provision_models}" == true ]] && ! check_bundled_models_remote; then
        missing=$((missing + 1))
    fi
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
warm_cargo_caches_remote
if [[ "${provision_models}" == true ]]; then
    provision_bundled_models_remote
fi
log "Done."
EOF
}

run_remote_bootstrap() {
    local -a workers=()

    if [[ -n "${TARGET_WORKER}" ]]; then
        workers=("$(resolve_worker_target "${TARGET_WORKER}")")
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

if [[ "${PROVISION_MODELS}" == true ]]; then
    validate_model_specs
fi

if [[ -n "${TARGET_WORKER}" || "${ALL_WORKERS}" == true ]]; then
    run_remote_bootstrap
else
    run_local_bootstrap "${MODE}"
fi
