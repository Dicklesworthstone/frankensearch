#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="audit"
SCOPE="workspace"
ROOT_PACKAGE="frankensearch"
METADATA_FILE=""
REGISTRY_CENSUS=""
RELEASE_TAG=""
OUTPUT_FILE=""
SOURCE_SHA_OVERRIDE=""
ALLOW_DIRTY=false
SCHEMA_VERSION="frankensearch-crates-publish-plan-v1"
CENSUS_SCHEMA_VERSION="frankensearch-crates-registry-census-v1"
USER_AGENT="frankensearch-publish-contract/1.0 (https://github.com/Dicklesworthstone/frankensearch)"

usage() {
  cat <<'USAGE'
Usage: scripts/check_crates_publish_contract.sh [OPTIONS]

Builds a credential-free, fail-closed crates.io publication plan from Cargo
metadata. The plan is a receipt; this script never publishes a crate.

Options:
  --mode <audit|gate|self-test>       audit emits blockers; gate fails on them
  --scope <workspace|facade>          all publishable members or facade closure
  --root-package <name>               facade/root package (default: frankensearch)
  --metadata <path>                   injected Cargo metadata JSON
  --registry-census <path|live>       injected census or live crates.io census
  --release-tag <tag>                 required by gate; crates-v<facade-version>
  --output <path>                     JSON receipt path
  --source-sha <40-hex>               candidate source identity override
  --allow-dirty                       test-only escape hatch for synthetic fixtures
  -h, --help                          show this help

Registry census schema:
  {
    "schema": "frankensearch-crates-registry-census-v1",
    "entries": [
      {
        "name": "crate-name",
        "version": "1.2.3",
        "status": "published|unpublished|unknown",
        "checksum": "sha256 or null",
        "source_sha": "40-hex git sha or null",
        "source_dirty": "boolean or null"
      }
    ]
  }
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --scope)
      SCOPE="${2:-}"
      shift 2
      ;;
    --root-package)
      ROOT_PACKAGE="${2:-}"
      shift 2
      ;;
    --metadata)
      METADATA_FILE="${2:-}"
      shift 2
      ;;
    --registry-census)
      REGISTRY_CENSUS="${2:-}"
      shift 2
      ;;
    --release-tag)
      RELEASE_TAG="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="${2:-}"
      shift 2
      ;;
    --source-sha)
      SOURCE_SHA_OVERRIDE="${2:-}"
      shift 2
      ;;
    --allow-dirty)
      ALLOW_DIRTY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  audit|gate|self-test) ;;
  *)
    echo "ERROR: invalid --mode '$MODE' (expected audit|gate|self-test)" >&2
    exit 2
    ;;
esac

case "$SCOPE" in
  workspace|facade) ;;
  *)
    echo "ERROR: invalid --scope '$SCOPE' (expected workspace|facade)" >&2
    exit 2
    ;;
esac

for tool in cargo git jq; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $tool" >&2
    exit 2
  fi
done

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    echo "ERROR: sha256sum or shasum is required" >&2
    return 2
  fi
}

run_self_test() {
  local temp_dir metadata_path census_path positive_receipt negative_metadata negative_census
  local negative_receipt script_path source_sha
  temp_dir="$(mktemp -d /tmp/frankensearch-publish-contract-self-test.XXXXXX)"
  metadata_path="${temp_dir}/metadata-positive.json"
  census_path="${temp_dir}/census-positive.json"
  positive_receipt="${temp_dir}/receipt-positive.json"
  negative_metadata="${temp_dir}/metadata-negative.json"
  negative_census="${temp_dir}/census-negative.json"
  negative_receipt="${temp_dir}/receipt-negative.json"
  script_path="${BASH_SOURCE[0]}"
  source_sha="1111111111111111111111111111111111111111"

  jq -n --arg root "$temp_dir" '{
    workspace_root: $root,
    packages: [
      {
        name: "frankensearch",
        version: "0.4.0",
        manifest_path: ($root + "/facade/Cargo.toml"),
        publish: null,
        description: "facade",
        license: "MIT",
        license_file: null,
        repository: "https://example.invalid/repo",
        readme: "README.md",
        dependencies: [
          {
            name: "frankensearch-quill",
            source: null,
            req: "^0.3.0",
            kind: null,
            optional: true,
            path: ($root + "/quill")
          }
        ]
      },
      {
        name: "frankensearch-quill",
        version: "0.3.0",
        manifest_path: ($root + "/quill/Cargo.toml"),
        publish: null,
        description: "quill",
        license: "MIT",
        license_file: null,
        repository: "https://example.invalid/repo",
        readme: "README.md",
        dependencies: [
          {
            name: "frankensearch-core",
            source: null,
            req: "^0.3.0",
            kind: null,
            optional: false,
            path: ($root + "/core")
          }
        ]
      },
      {
        name: "frankensearch-core",
        version: "0.3.0",
        manifest_path: ($root + "/core/Cargo.toml"),
        publish: null,
        description: "core",
        license: "MIT",
        license_file: null,
        repository: "https://example.invalid/repo",
        readme: "README.md",
        dependencies: []
      }
    ]
  }' >"$metadata_path"

  jq -n --arg schema "$CENSUS_SCHEMA_VERSION" '{
    schema: $schema,
    entries: [
      {name: "frankensearch-core", version: "0.3.0", status: "unpublished", checksum: null, source_sha: null, source_dirty: null},
      {name: "frankensearch-quill", version: "0.3.0", status: "unpublished", checksum: null, source_sha: null, source_dirty: null},
      {name: "frankensearch", version: "0.4.0", status: "unpublished", checksum: null, source_sha: null, source_dirty: null}
    ]
  }' >"$census_path"

  bash "$script_path" \
    --mode gate \
    --scope facade \
    --metadata "$metadata_path" \
    --registry-census "$census_path" \
    --release-tag "crates-v0.4.0" \
    --source-sha "$source_sha" \
    --allow-dirty \
    --output "$positive_receipt"

  jq -e '
    .status == "ready"
    and [.packages[].name] == [
      "frankensearch-core",
      "frankensearch-quill",
      "frankensearch"
    ]
    and (.blockers | length) == 0
  ' "$positive_receipt" >/dev/null

  jq --arg root "$temp_dir" '
    (.packages[] | select(.name == "frankensearch-quill") | .readme) = null
    | (.packages[] | select(.name == "frankensearch-quill") | .dependencies[] | select(.name == "frankensearch-core") | .req) = "^9.9.9"
    | (.packages[] | select(.name == "frankensearch-quill") | .dependencies) += [
        {
          name: "git-only-engine",
          source: "git+https://example.invalid/engine?rev=abc",
          req: "*",
          kind: null,
          optional: true,
          path: null
        }
      ]
    | (.packages[] | select(.name == "frankensearch-core") | .dependencies) += [
        {
          name: "frankensearch-quill",
          source: null,
          req: "*",
          kind: null,
          optional: false,
          path: ($root + "/quill")
        }
      ]
  ' "$metadata_path" >"$negative_metadata"

  jq --arg source_sha "2222222222222222222222222222222222222222" '
    (.entries[] | select(.name == "frankensearch-core")) = {
      name: "frankensearch-core",
      version: "0.3.0",
      status: "published",
      checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      source_sha: $source_sha,
      source_dirty: false
    }
  ' "$census_path" >"$negative_census"

  if bash "$script_path" \
    --mode gate \
    --scope facade \
    --metadata "$negative_metadata" \
    --registry-census "$negative_census" \
    --release-tag "crates-v0.4.0" \
    --source-sha "$source_sha" \
    --allow-dirty \
    --output "$negative_receipt"; then
    echo "ERROR: negative self-test unexpectedly passed" >&2
    return 1
  fi

  jq -e '
    ([.blockers[].code] | index("PACKAGE_README_MISSING")) != null
    and ([.blockers[].code] | index("DEPENDENCY_GIT_VERSION_REQUIRED")) != null
    and ([.blockers[].code] | index("INTERNAL_DEPENDENCY_CYCLE")) != null
    and ([.blockers[].code] | index("INTERNAL_DEPENDENCY_VERSION_MISMATCH")) != null
    and ([.blockers[].code] | index("INTERNAL_DEPENDENCY_VERSION_REQUIRED")) != null
    and ([.blockers[].code] | index("PUBLISHED_VERSION_SOURCE_MISMATCH")) != null
  ' "$negative_receipt" >/dev/null

  echo "publish-contract self-test passed"
  echo "positive receipt: $positive_receipt"
  echo "negative receipt: $negative_receipt"
}

if [[ "$MODE" == "self-test" ]]; then
  run_self_test
  exit 0
fi

TEMP_DIR="$(mktemp -d /tmp/frankensearch-publish-contract.XXXXXX)"
BLOCKERS_FILE="${TEMP_DIR}/blockers.jsonl"
PACKAGES_FILE="${TEMP_DIR}/packages.jsonl"
: >"$BLOCKERS_FILE"
: >"$PACKAGES_FILE"

add_blocker() {
  local code="$1"
  local package_name="$2"
  local dependency_name="$3"
  local bead_id="$4"
  local message="$5"
  local remediation="$6"

  jq -cn \
    --arg code "$code" \
    --arg package "$package_name" \
    --arg dependency "$dependency_name" \
    --arg bead "$bead_id" \
    --arg message "$message" \
    --arg remediation "$remediation" \
    '{
      code: $code,
      package: (if $package == "" then null else $package end),
      dependency: (if $dependency == "" then null else $dependency end),
      bead: (if $bead == "" then null else $bead end),
      message: $message,
      remediation: $remediation
    }' >>"$BLOCKERS_FILE"
}

dependency_bead() {
  case "$1" in
    hnsw_rs) echo "bd-mczj" ;;
    ft-api|ft-autograd|ft-core) echo "bd-8nqz.6-ft-registry" ;;
    *) echo "bd-8nqz.6" ;;
  esac
}

if [[ -z "$METADATA_FILE" ]]; then
  METADATA_FILE="${TEMP_DIR}/cargo-metadata.json"
  (
    cd "$ROOT_DIR"
    cargo metadata --locked --no-deps --format-version 1
  ) >"$METADATA_FILE"
fi

if [[ ! -f "$METADATA_FILE" ]]; then
  echo "ERROR: metadata file not found: $METADATA_FILE" >&2
  exit 2
fi
if ! jq -e '.workspace_root and (.packages | type == "array")' "$METADATA_FILE" >/dev/null; then
  echo "ERROR: invalid Cargo metadata JSON: $METADATA_FILE" >&2
  exit 2
fi

WORKSPACE_ROOT="$(jq -r '.workspace_root' "$METADATA_FILE")"
if [[ -z "$WORKSPACE_ROOT" || "$WORKSPACE_ROOT" == "null" ]]; then
  echo "ERROR: Cargo metadata omitted workspace_root" >&2
  exit 2
fi

if ! jq -e --arg package "$ROOT_PACKAGE" 'any(.packages[]; .name == $package)' "$METADATA_FILE" >/dev/null; then
  echo "ERROR: root package '$ROOT_PACKAGE' is absent from metadata" >&2
  exit 2
fi

if [[ -n "$SOURCE_SHA_OVERRIDE" ]]; then
  SOURCE_SHA="$SOURCE_SHA_OVERRIDE"
else
  SOURCE_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD)"
fi
if [[ ! "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: candidate source SHA must be exactly 40 lowercase hex characters" >&2
  exit 2
fi

SOURCE_DIRTY=false
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=no)" ]]; then
  SOURCE_DIRTY=true
fi
if [[ "$SOURCE_DIRTY" == true && "$ALLOW_DIRTY" == false ]]; then
  add_blocker \
    "DIRTY_TRACKED_WORKTREE" \
    "" \
    "" \
    "bd-8nqz.6" \
    "Tracked files differ from candidate source SHA ${SOURCE_SHA}." \
    "Run the gate from a clean checkout at the exact release commit."
fi

LOCKFILE="${ROOT_DIR}/Cargo.lock"
LOCKFILE_SHA=""
if [[ -f "$LOCKFILE" ]]; then
  LOCKFILE_SHA="$(sha256_file "$LOCKFILE")"
else
  add_blocker \
    "CARGO_LOCK_MISSING" \
    "" \
    "" \
    "bd-8nqz.6" \
    "Cargo.lock is missing from the publication candidate." \
    "Generate and commit the workspace lockfile before release planning."
fi

mapfile -t WORKSPACE_PACKAGES < <(jq -r '.packages[].name' "$METADATA_FILE" | LC_ALL=C sort -u)
declare -A IS_WORKSPACE=()
for package_name in "${WORKSPACE_PACKAGES[@]}"; do
  IS_WORKSPACE["$package_name"]=1
done

package_is_publishable() {
  local package_name="$1"
  jq -e --arg package "$package_name" '
    any(
      .packages[];
      .name == $package
      and (.publish == null or (.publish | length) > 0)
    )
  ' "$METADATA_FILE" >/dev/null
}

package_internal_dependencies() {
  local package_name="$1"
  jq -r --arg package "$package_name" '
    .packages[]
    | select(.name == $package)
    | .dependencies[]
    | select((.kind // "normal") != "dev")
    | .name
  ' "$METADATA_FILE" | LC_ALL=C sort -u
}

declare -A IN_SCOPE=()
if [[ "$SCOPE" == "workspace" ]]; then
  for package_name in "${WORKSPACE_PACKAGES[@]}"; do
    if package_is_publishable "$package_name"; then
      IN_SCOPE["$package_name"]=1
    fi
  done
else
  QUEUE=("$ROOT_PACKAGE")
  queue_index=0
  while (( queue_index < ${#QUEUE[@]} )); do
    package_name="${QUEUE[$queue_index]}"
    queue_index=$((queue_index + 1))
    if [[ -n "${IN_SCOPE[$package_name]:-}" ]]; then
      continue
    fi
    IN_SCOPE["$package_name"]=1
    while IFS= read -r dependency_name; do
      [[ -z "$dependency_name" ]] && continue
      if [[ -n "${IS_WORKSPACE[$dependency_name]:-}" ]]; then
        QUEUE+=("$dependency_name")
      fi
    done < <(package_internal_dependencies "$package_name")
  done
fi

mapfile -t SCOPE_PACKAGES < <(printf '%s\n' "${!IN_SCOPE[@]}" | LC_ALL=C sort)
for package_name in "${SCOPE_PACKAGES[@]}"; do
  if ! package_is_publishable "$package_name"; then
    add_blocker \
      "INTERNAL_DEPENDENCY_NOT_PUBLISHABLE" \
      "$package_name" \
      "" \
      "bd-8nqz.6" \
      "Package '$package_name' is required by the selected scope but has publish = false." \
      "Publish the dependency or remove it from the registry-facing dependency graph."
  fi
done

declare -A ORDERED=()
SEQUENCE=()
while (( ${#SEQUENCE[@]} < ${#SCOPE_PACKAGES[@]} )); do
  progress=false
  for package_name in "${SCOPE_PACKAGES[@]}"; do
    [[ -n "${ORDERED[$package_name]:-}" ]] && continue
    if ! package_is_publishable "$package_name"; then
      ORDERED["$package_name"]=1
      SEQUENCE+=("$package_name")
      progress=true
      break
    fi

    ready=true
    while IFS= read -r dependency_name; do
      [[ -z "$dependency_name" ]] && continue
      if [[ -n "${IN_SCOPE[$dependency_name]:-}" && -z "${ORDERED[$dependency_name]:-}" ]]; then
        ready=false
        break
      fi
    done < <(package_internal_dependencies "$package_name")

    if [[ "$ready" == true ]]; then
      ORDERED["$package_name"]=1
      SEQUENCE+=("$package_name")
      progress=true
      break
    fi
  done

  if [[ "$progress" == false ]]; then
    mapfile -t cycle_members < <(
      for package_name in "${SCOPE_PACKAGES[@]}"; do
        if [[ -z "${ORDERED[$package_name]:-}" ]]; then
          printf '%s\n' "$package_name"
        fi
      done
    )
    cycle_csv="$(IFS=,; echo "${cycle_members[*]}")"
    add_blocker \
      "INTERNAL_DEPENDENCY_CYCLE" \
      "" \
      "" \
      "bd-8nqz.6" \
      "No topological publication order exists for: ${cycle_csv}." \
      "Remove the package cycle before attempting crates.io publication."
    for package_name in "${cycle_members[@]}"; do
      ORDERED["$package_name"]=1
      SEQUENCE+=("$package_name")
    done
  fi
done

for package_name in "${SEQUENCE[@]}"; do
  package_json="$(jq -c --arg package "$package_name" '.packages[] | select(.name == $package)' "$METADATA_FILE")"
  description="$(jq -r '.description // ""' <<<"$package_json")"
  license="$(jq -r '.license // ""' <<<"$package_json")"
  license_file="$(jq -r '.license_file // ""' <<<"$package_json")"
  repository="$(jq -r '.repository // ""' <<<"$package_json")"
  readme="$(jq -r '.readme // ""' <<<"$package_json")"

  if [[ -z "$description" ]]; then
    add_blocker \
      "PACKAGE_DESCRIPTION_MISSING" \
      "$package_name" \
      "" \
      "bd-8nqz.6" \
      "Package '$package_name' has no crates.io description." \
      "Add package.description before publication."
  fi
  if [[ -z "$license" && -z "$license_file" ]]; then
    add_blocker \
      "PACKAGE_LICENSE_MISSING" \
      "$package_name" \
      "" \
      "bd-8nqz.6" \
      "Package '$package_name' declares neither license nor license-file." \
      "Add an SPDX license expression or inherited license-file."
  fi
  if [[ -z "$repository" ]]; then
    add_blocker \
      "PACKAGE_REPOSITORY_MISSING" \
      "$package_name" \
      "" \
      "bd-8nqz.6" \
      "Package '$package_name' has no repository URL." \
      "Add package.repository before publication."
  fi
  if [[ -z "$readme" ]]; then
    add_blocker \
      "PACKAGE_README_MISSING" \
      "$package_name" \
      "" \
      "bd-8nqz.6" \
      "Package '$package_name' has no README in its package metadata." \
      "Add and declare a package README before publication."
  fi

  while IFS= read -r dependency_json; do
    [[ -z "$dependency_json" ]] && continue
    dependency_name="$(jq -r '.name' <<<"$dependency_json")"
    dependency_source="$(jq -r '.source // ""' <<<"$dependency_json")"
    dependency_req="$(jq -r '.req // ""' <<<"$dependency_json")"

    if [[ -n "${IS_WORKSPACE[$dependency_name]:-}" ]]; then
      if [[ -z "$dependency_req" || "$dependency_req" == "*" ]]; then
        add_blocker \
          "INTERNAL_DEPENDENCY_VERSION_REQUIRED" \
          "$package_name" \
          "$dependency_name" \
          "bd-8nqz.6" \
          "Internal dependency '$dependency_name' in '$package_name' has no publishable version requirement." \
          "Add a version requirement matching the candidate dependency version."
      else
        dependency_version="$(
          jq -r --arg package "$dependency_name" \
            '.packages[] | select(.name == $package) | .version' \
            "$METADATA_FILE"
        )"
        case "$dependency_req" in
          "$dependency_version"|"^$dependency_version"|"=$dependency_version") ;;
          *)
            add_blocker \
              "INTERNAL_DEPENDENCY_VERSION_MISMATCH" \
              "$package_name" \
              "$dependency_name" \
              "bd-8nqz.6" \
              "Internal dependency '$dependency_name' in '$package_name' requires '$dependency_req', not candidate '$dependency_version'." \
              "Use the exact candidate version or its caret requirement."
            ;;
        esac
      fi
      if ! package_is_publishable "$dependency_name"; then
        add_blocker \
          "INTERNAL_DEPENDENCY_NOT_PUBLISHABLE" \
          "$package_name" \
          "$dependency_name" \
          "bd-8nqz.6" \
          "Registry package '$package_name' depends on unpublished workspace package '$dependency_name'." \
          "Publish the dependency or remove it from the registry-facing graph."
      fi
    elif [[ "$dependency_source" == git+* ]]; then
      if [[ -z "$dependency_req" || "$dependency_req" == "*" ]]; then
        bead_id="$(dependency_bead "$dependency_name")"
        add_blocker \
          "DEPENDENCY_GIT_VERSION_REQUIRED" \
          "$package_name" \
          "$dependency_name" \
          "$bead_id" \
          "Git dependency '$dependency_name' in '$package_name' has no registry version requirement." \
          "Publish a registry-equivalent crate and add its concrete version, or remove the dependency."
      fi
    fi
  done < <(
    jq -c '
      .dependencies[]
      | select((.kind // "normal") != "dev")
    ' <<<"$package_json"
  )
done

ROOT_VERSION="$(jq -r --arg package "$ROOT_PACKAGE" '.packages[] | select(.name == $package) | .version' "$METADATA_FILE")"
if [[ -n "$RELEASE_TAG" ]]; then
  expected_tag="crates-v${ROOT_VERSION}"
  if [[ "$RELEASE_TAG" != "$expected_tag" ]]; then
    add_blocker \
      "RELEASE_TAG_MISMATCH" \
      "$ROOT_PACKAGE" \
      "" \
      "bd-8nqz.6" \
      "Crate-bundle tag '$RELEASE_TAG' does not match expected '$expected_tag'." \
      "Use the crates-v<facade-version> namespace; package versions remain heterogeneous in the receipt."
  fi
elif [[ "$MODE" == "gate" ]]; then
  add_blocker \
    "RELEASE_TAG_REQUIRED" \
    "$ROOT_PACKAGE" \
    "" \
    "bd-8nqz.6" \
    "Gate mode requires a content-addressed crate-bundle tag." \
    "Pass --release-tag crates-v${ROOT_VERSION}."
fi

generate_live_census() {
  local output_path="$1"
  local entries_file="${TEMP_DIR}/live-census-entries.jsonl"
  : >"$entries_file"

  if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
    add_blocker \
      "REGISTRY_CENSUS_TOOL_MISSING" \
      "" \
      "" \
      "bd-8nqz.6" \
      "Live registry census requires curl and tar." \
      "Install the tools or inject a signed census JSON file."
    jq -n --arg schema "$CENSUS_SCHEMA_VERSION" '{schema: $schema, entries: []}' >"$output_path"
    return
  fi

  for package_name in "${SEQUENCE[@]}"; do
    package_version="$(jq -r --arg package "$package_name" '.packages[] | select(.name == $package) | .version' "$METADATA_FILE")"
    api_response="${TEMP_DIR}/${package_name}-${package_version}-api.json"
    archive_path="${TEMP_DIR}/${package_name}-${package_version}.crate"
    http_status="$(
      curl \
        --silent \
        --show-error \
        --location \
        --retry 3 \
        --retry-all-errors \
        --retry-delay 2 \
        --user-agent "$USER_AGENT" \
        --output "$api_response" \
        --write-out '%{http_code}' \
        "https://crates.io/api/v1/crates/${package_name}/${package_version}" || true
    )"

    if [[ "$http_status" == "404" ]]; then
      jq -cn \
        --arg name "$package_name" \
        --arg version "$package_version" \
        '{
          name: $name,
          version: $version,
          status: "unpublished",
          checksum: null,
          source_sha: null,
          source_dirty: null
        }' >>"$entries_file"
      continue
    fi

    if [[ "$http_status" != "200" ]]; then
      add_blocker \
        "REGISTRY_CENSUS_QUERY_FAILED" \
        "$package_name" \
        "" \
        "bd-8nqz.6" \
        "crates.io census query returned HTTP ${http_status:-unknown} for ${package_name} ${package_version}." \
        "Retry from a networked worker or inject a complete census receipt."
      jq -cn \
        --arg name "$package_name" \
        --arg version "$package_version" \
        '{
          name: $name,
          version: $version,
          status: "unknown",
          checksum: null,
          source_sha: null,
          source_dirty: null
        }' >>"$entries_file"
      continue
    fi

    registry_checksum="$(jq -r '.version.checksum // ""' "$api_response")"
    download_status="$(
      curl \
        --silent \
        --show-error \
        --location \
        --retry 3 \
        --retry-all-errors \
        --retry-delay 2 \
        --user-agent "$USER_AGENT" \
        --output "$archive_path" \
        --write-out '%{http_code}' \
        "https://crates.io/api/v1/crates/${package_name}/${package_version}/download" || true
    )"
    if [[ "$download_status" != "200" ]]; then
      add_blocker \
        "REGISTRY_ARCHIVE_DOWNLOAD_FAILED" \
        "$package_name" \
        "" \
        "bd-8nqz.6" \
        "Published archive download returned HTTP ${download_status:-unknown} for ${package_name} ${package_version}." \
        "Retry from a networked worker or inject a complete census receipt."
      jq -cn \
        --arg name "$package_name" \
        --arg version "$package_version" \
        --arg checksum "$registry_checksum" \
        '{
          name: $name,
          version: $version,
          status: "unknown",
          checksum: (if $checksum == "" then null else $checksum end),
          source_sha: null,
          source_dirty: null
        }' >>"$entries_file"
      continue
    fi

    archive_checksum="$(sha256_file "$archive_path")"
    if [[ -z "$registry_checksum" || "$archive_checksum" != "$registry_checksum" ]]; then
      add_blocker \
        "REGISTRY_ARCHIVE_CHECKSUM_MISMATCH" \
        "$package_name" \
        "" \
        "bd-8nqz.6" \
        "Downloaded archive checksum does not match crates.io metadata for ${package_name} ${package_version}." \
        "Stop publication and investigate registry or transport integrity."
    fi

    vcs_info_path="$(
      tar -tzf "$archive_path" |
        awk '/\/[.]cargo_vcs_info[.]json$/ && first == "" {first = $0} END {print first}'
    )"
    published_source_sha=""
    published_source_dirty=""
    if [[ -n "$vcs_info_path" ]]; then
      vcs_info_json="$(tar -xOzf "$archive_path" "$vcs_info_path")"
      published_source_sha="$(jq -r '.git.sha1 // ""' <<<"$vcs_info_json")"
      published_source_dirty="$(jq -r '.git.dirty // false' <<<"$vcs_info_json")"
    fi
    if [[ ! "$published_source_sha" =~ ^[0-9a-f]{40}$ ]]; then
      add_blocker \
        "PUBLISHED_PROVENANCE_MISSING" \
        "$package_name" \
        "" \
        "bd-8nqz.6" \
        "Published ${package_name} ${package_version} lacks a usable .cargo_vcs_info.json source SHA." \
        "Treat the version as occupied and choose a new version."
    fi

    jq -cn \
      --arg name "$package_name" \
      --arg version "$package_version" \
      --arg checksum "$archive_checksum" \
      --arg source_sha "$published_source_sha" \
      --argjson source_dirty "${published_source_dirty:-null}" \
      '{
        name: $name,
        version: $version,
        status: "published",
        checksum: $checksum,
        source_sha: (if $source_sha == "" then null else $source_sha end),
        source_dirty: $source_dirty
      }' >>"$entries_file"
  done

  jq -s \
    --arg schema "$CENSUS_SCHEMA_VERSION" \
    --arg endpoint "https://crates.io/api/v1/crates" \
    '{
      schema: $schema,
      registry: "crates.io",
      endpoint: $endpoint,
      entries: .
    }' "$entries_file" >"$output_path"
}

CENSUS_FILE=""
if [[ "$REGISTRY_CENSUS" == "live" ]]; then
  CENSUS_FILE="${TEMP_DIR}/registry-census-live.json"
  generate_live_census "$CENSUS_FILE"
elif [[ -n "$REGISTRY_CENSUS" ]]; then
  CENSUS_FILE="$REGISTRY_CENSUS"
fi

if [[ -n "$CENSUS_FILE" ]]; then
  if [[ ! -f "$CENSUS_FILE" ]]; then
    echo "ERROR: registry census file not found: $CENSUS_FILE" >&2
    exit 2
  fi
  if ! jq -e \
    --arg schema "$CENSUS_SCHEMA_VERSION" \
    '.schema == $schema and (.entries | type == "array")' \
    "$CENSUS_FILE" >/dev/null; then
    echo "ERROR: invalid registry census JSON: $CENSUS_FILE" >&2
    exit 2
  fi

  duplicate_registry_entries="$(
    jq -r '
      .entries
      | group_by([.name, .version])
      | .[]
      | select(length > 1)
      | "\(.[0].name) \(.[0].version)"
    ' "$CENSUS_FILE"
  )"
  if [[ -n "$duplicate_registry_entries" ]]; then
    add_blocker \
      "REGISTRY_CENSUS_DUPLICATE" \
      "" \
      "" \
      "bd-8nqz.6" \
      "Registry census contains duplicate crate/version entries: ${duplicate_registry_entries//$'\n'/, }." \
      "Regenerate a one-row-per-crate-version census."
  fi
elif [[ "$MODE" == "gate" ]]; then
  add_blocker \
    "REGISTRY_CENSUS_REQUIRED" \
    "" \
    "" \
    "bd-8nqz.6" \
    "Gate mode has no registry census, so occupied versions cannot be authenticated." \
    "Pass --registry-census live or an independently captured census JSON."
fi

for position in "${!SEQUENCE[@]}"; do
  package_name="${SEQUENCE[$position]}"
  package_json="$(jq -c --arg package "$package_name" '.packages[] | select(.name == $package)' "$METADATA_FILE")"
  package_version="$(jq -r '.version' <<<"$package_json")"
  manifest_path="$(jq -r '.manifest_path' <<<"$package_json")"
  relative_manifest="$manifest_path"
  if [[ "$manifest_path" == "$WORKSPACE_ROOT/"* ]]; then
    relative_manifest="${manifest_path#"$WORKSPACE_ROOT/"}"
  fi

  mapfile -t internal_dependencies < <(
    package_internal_dependencies "$package_name" |
      while IFS= read -r dependency_name; do
        if [[ -n "${IN_SCOPE[$dependency_name]:-}" ]]; then
          printf '%s\n' "$dependency_name"
        fi
      done
  )
  if (( ${#internal_dependencies[@]} == 0 )); then
    dependencies_json='[]'
  else
    dependencies_json="$(printf '%s\n' "${internal_dependencies[@]}" | jq -R -s 'split("\n") | map(select(length > 0))')"
  fi

  registry_status="not_checked"
  registry_checksum=""
  registry_source_sha=""
  registry_source_dirty=""
  action="unknown"
  if [[ -n "$CENSUS_FILE" ]]; then
    entry_count="$(
      jq --arg name "$package_name" --arg version "$package_version" \
        '[.entries[] | select(.name == $name and .version == $version)] | length' \
        "$CENSUS_FILE"
    )"
    if [[ "$entry_count" != "1" ]]; then
      add_blocker \
        "REGISTRY_CENSUS_ENTRY_MISSING" \
        "$package_name" \
        "" \
        "bd-8nqz.6" \
        "Registry census does not contain exactly one entry for ${package_name} ${package_version}." \
        "Regenerate a complete census for the exact publication plan."
      registry_status="unknown"
      action="blocked"
    else
      registry_status="$(
        jq -r --arg name "$package_name" --arg version "$package_version" \
          '.entries[] | select(.name == $name and .version == $version) | .status' \
          "$CENSUS_FILE"
      )"
      registry_checksum="$(
        jq -r --arg name "$package_name" --arg version "$package_version" \
          '.entries[] | select(.name == $name and .version == $version) | .checksum // ""' \
          "$CENSUS_FILE"
      )"
      registry_source_sha="$(
        jq -r --arg name "$package_name" --arg version "$package_version" \
          '.entries[] | select(.name == $name and .version == $version) | .source_sha // ""' \
          "$CENSUS_FILE"
      )"
      registry_source_dirty="$(
        jq -r --arg name "$package_name" --arg version "$package_version" \
          '.entries[] | select(.name == $name and .version == $version) | .source_dirty // false' \
          "$CENSUS_FILE"
      )"

      case "$registry_status" in
        unpublished)
          action="publish"
          ;;
        published)
          if [[ "$registry_source_dirty" == "true" ]]; then
            action="blocked_dirty_publication"
            add_blocker \
              "PUBLISHED_VERSION_DIRTY_SOURCE" \
              "$package_name" \
              "" \
              "bd-8nqz.6" \
              "Version ${package_name} ${package_version} was published from a dirty worktree." \
              "Treat the version as occupied and choose a new version."
          elif [[ "$registry_source_sha" == "$SOURCE_SHA" ]]; then
            candidate_target_dir="${TEMP_DIR}/candidate-package-target"
            package_log="${TEMP_DIR}/${package_name}-${package_version}-package.log"
            if (
              cd "$ROOT_DIR"
              CARGO_TARGET_DIR="$candidate_target_dir" \
                cargo package \
                  --locked \
                  --no-verify \
                  -p "$package_name"
            ) >"$package_log" 2>&1; then
              candidate_archive="${candidate_target_dir}/package/${package_name}-${package_version}.crate"
              if [[ ! -f "$candidate_archive" ]]; then
                action="blocked"
                add_blocker \
                  "CANDIDATE_PACKAGE_ARCHIVE_MISSING" \
                  "$package_name" \
                  "" \
                  "bd-8nqz.6" \
                  "Cargo succeeded but emitted no candidate archive for ${package_name} ${package_version}." \
                  "Inspect ${package_log} and Cargo target-directory handling."
              else
                candidate_checksum="$(sha256_file "$candidate_archive")"
                if [[ -n "$registry_checksum" && "$candidate_checksum" == "$registry_checksum" ]]; then
                  action="skip_exact_archive"
                else
                  action="blocked_content_mismatch"
                  add_blocker \
                    "PUBLISHED_VERSION_CONTENT_MISMATCH" \
                    "$package_name" \
                    "" \
                    "bd-8nqz.6" \
                    "Version ${package_name} ${package_version} names the candidate source SHA but its archive checksum differs." \
                    "Treat the version as occupied and choose a new version."
                fi
              fi
            else
              action="blocked"
              add_blocker \
                "CANDIDATE_PACKAGE_CHECKSUM_FAILED" \
                "$package_name" \
                "" \
                "bd-8nqz.6" \
                "Could not build the candidate archive needed to authenticate existing ${package_name} ${package_version}." \
                "Inspect ${package_log}; never accept source SHA alone as exact package identity."
            fi
          else
            action="blocked_version_reuse"
            add_blocker \
              "PUBLISHED_VERSION_SOURCE_MISMATCH" \
              "$package_name" \
              "" \
              "bd-8nqz.6" \
              "Version ${package_name} ${package_version} is already published from source ${registry_source_sha:-unknown}, not candidate ${SOURCE_SHA}." \
              "Bump the crate version and every dependent internal requirement."
          fi
          ;;
        unknown)
          action="blocked"
          add_blocker \
            "REGISTRY_CENSUS_STATUS_UNKNOWN" \
            "$package_name" \
            "" \
            "bd-8nqz.6" \
            "Registry status is unknown for ${package_name} ${package_version}." \
            "Repeat or independently supply the registry census."
          ;;
        *)
          action="blocked"
          add_blocker \
            "REGISTRY_CENSUS_STATUS_INVALID" \
            "$package_name" \
            "" \
            "bd-8nqz.6" \
            "Registry census uses invalid status '$registry_status' for ${package_name} ${package_version}." \
            "Use published, unpublished, or unknown."
          ;;
      esac
    fi
  fi

  jq -cn \
    --argjson position "$position" \
    --arg name "$package_name" \
    --arg version "$package_version" \
    --arg manifest "$relative_manifest" \
    --argjson dependencies "$dependencies_json" \
    --arg registry_status "$registry_status" \
    --arg registry_checksum "$registry_checksum" \
    --arg registry_source_sha "$registry_source_sha" \
    --arg registry_source_dirty "$registry_source_dirty" \
    --arg action "$action" \
    '{
      position: $position,
      name: $name,
      version: $version,
      manifest: $manifest,
      internal_dependencies: $dependencies,
      registry: {
        status: $registry_status,
        checksum: (if $registry_checksum == "" then null else $registry_checksum end),
        source_sha: (if $registry_source_sha == "" then null else $registry_source_sha end),
        source_dirty: (
          if $registry_source_dirty == ""
          then null
          else ($registry_source_dirty == "true")
          end
        )
      },
      action: $action
    }' >>"$PACKAGES_FILE"
done

if [[ -z "$OUTPUT_FILE" ]]; then
  OUTPUT_DIR="/tmp/frankensearch-crates-publish-contract/${SOURCE_SHA:0:12}"
  mkdir -p "$OUTPUT_DIR"
  OUTPUT_FILE="${OUTPUT_DIR}/publish-plan.json"
else
  mkdir -p "$(dirname "$OUTPUT_FILE")"
fi

BLOCKERS_JSON="$(jq -s 'sort_by(.code, .package, .dependency) | unique' "$BLOCKERS_FILE")"
PACKAGES_JSON="$(jq -s 'sort_by(.position)' "$PACKAGES_FILE")"
BLOCKER_COUNT="$(jq 'length' <<<"$BLOCKERS_JSON")"
if (( BLOCKER_COUNT == 0 )); then
  if [[ -n "$CENSUS_FILE" ]]; then
    STATUS="ready"
  else
    STATUS="incomplete"
  fi
else
  STATUS="blocked"
fi

RUSTC_VERSION="$(rustc -V)"
CARGO_VERSION="$(cargo -V)"
GENERATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

jq -n \
  --arg schema "$SCHEMA_VERSION" \
  --arg mode "$MODE" \
  --arg scope "$SCOPE" \
  --arg status "$STATUS" \
  --arg root_package "$ROOT_PACKAGE" \
  --arg root_version "$ROOT_VERSION" \
  --arg release_tag "$RELEASE_TAG" \
  --arg source_sha "$SOURCE_SHA" \
  --argjson source_dirty "$SOURCE_DIRTY" \
  --arg lockfile_sha256 "$LOCKFILE_SHA" \
  --arg rustc "$RUSTC_VERSION" \
  --arg cargo "$CARGO_VERSION" \
  --arg generated_at "$GENERATED_AT" \
  --argjson packages "$PACKAGES_JSON" \
  --argjson blockers "$BLOCKERS_JSON" \
  '{
    schema: $schema,
    mode: $mode,
    scope: $scope,
    status: $status,
    root_package: {
      name: $root_package,
      version: $root_version
    },
    release: {
      tag: (if $release_tag == "" then null else $release_tag end),
      tag_namespace: "crates-v<facade-version>",
      heterogeneous_package_versions: true
    },
    source: {
      git_sha: $source_sha,
      tracked_worktree_dirty: $source_dirty,
      cargo_lock_sha256: (if $lockfile_sha256 == "" then null else $lockfile_sha256 end),
      rustc: $rustc,
      cargo: $cargo
    },
    generated_at: $generated_at,
    packages: $packages,
    blockers: $blockers,
    blocker_codes: ($blockers | map(.code) | unique)
  }' >"$OUTPUT_FILE"

echo "publish-contract status: $STATUS"
echo "publish scope: $SCOPE (${#SEQUENCE[@]} packages)"
echo "receipt: $OUTPUT_FILE"
if (( BLOCKER_COUNT > 0 )); then
  jq -r '.[] | "BLOCKED [\(.code)] \(.message)"' <<<"$BLOCKERS_JSON" >&2
fi

if [[ "$MODE" == "gate" && "$STATUS" != "ready" ]]; then
  exit 1
fi
