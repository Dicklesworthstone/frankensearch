#!/usr/bin/env bash
# Run the real installer with local stand-ins for the source build and candidate.
set -euo pipefail
installer=$(cd "$(dirname "$0")/../.." && pwd)/install.sh
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/tools" "$work/staging files" "$work/dest"
export FSFS_INSTALL_MIN_DEST_MB=1 FSFS_INSTALL_MIN_STAGE_SOURCE_MB=1
export FSFS_TEST_WORK="$work" RUSTUP_INIT_SKIP=1 NO_COLOR=1
export TMPDIR="$work/staging files" FSFS_INSTALL_LOCK_FILE="$work/install.lock"
unset VERSION ARTIFACT_URL CHECKSUM CHECKSUM_URL
cat >"$work/tools/uname" <<'SH'
#!/bin/sh
case "$1" in -s) echo "${FSFS_TEST_OS:-Darwin}";; -m) echo x86_64;; *) exit 1;; esac
SH
cat >"$work/tools/git" <<'SH'
#!/usr/bin/env bash
set -eu
[[ "$1" == clone && "$*" == *'--branch v1.13.0 '* ]]
mkdir -p "${!#}"
SH
cat >"$work/tools/cargo" <<'SH'
#!/bin/sh
set -eu
[ "$*" = 'build --release -p frankensearch-fsfs --no-default-features --features semantic-native' ]
mkdir -p target/release
cp "$FSFS_TEST_WORK/candidate" target/release/fsfs
SH
cat >"$work/candidate" <<'SH'
#!/usr/bin/env bash
set -eu
echo "$*" >> "$FSFS_TEST_WORK/calls"
case "$1" in
  version) echo 'fsfs 1.13.0' ;;
  help)
    case "$FSFS_TEST_CASE" in loader*) echo 'download-models --verify-loader';; *) echo 'fsfs doctor';; esac ;;
  download-models)
    quality=all-minilm-l6-v2-native
    if [[ "${FSFS_TEST_STANDARD:-0}" == 1 ]]; then quality=all-minilm-l6-v2; fi
    [ "$2" = potion-multilingual-128m ] || [ "$2" = "$quality" ]
    if [[ "${3:-}" == --verify-loader ]]; then
      [[ "$4" == --offline ]]
      [[ "$FSFS_TEST_CASE" != loader-fail || "$2" != all-minilm-l6-v2-native ]]
    elif [[ "${4:-}" == --format ]]; then
      [[ "$3 $4 $5" == '--verify --format json' && "$2" == all-minilm-l6-v2-native ]]
      printf '{"ok":true,"data":{"model_root":"%s/custom model cache","models":[{"id":"all-minilm-l6-v2-native","verified":true}]}}\n' "$FSFS_TEST_WORK"
    else
      [[ "$#" == 2 || "$3" == --verify ]]
    fi ;;
  doctor)
    [[ "$2 $3 $4 $5" == '--offline --format json --index-dir' ]]
    [[ "$6" == "$TMPDIR/"* && -d "$6" && -z "$(ls -A "$6")" ]]
    [[ "$7" == --config && "$9" == --no-fast-only ]]
    [[ "$FRANKENSEARCH_MODEL_DIR" == "$FSFS_TEST_WORK/custom model cache" ]]
    grep -Fxq 'quality_model = "all-minilm-l6-v2-native"' "$8"
    grep -Fxq 'fast_model = "potion-multilingual-128m"' "$8"
    case "$FSFS_TEST_CASE" in
      failure) exit 1;;
      envelope) echo '{"ok":false,"error":{"code":"doctor_failed"}}'; exit;;
    esac
    quality=all-minilm-l6-v2-native
    [[ "$FSFS_TEST_CASE" != wrong-model ]] || quality=all-minilm-l6-v2
    verdict=pass
    [[ "$FSFS_TEST_CASE" != warn ]] || verdict=warn
    printf '{"ok":true,"data":{"overall":"warn","checks":[{"name":"model.fast","verdict":"pass","detail":"potion-multilingual-128M is manifest-verified and loadable at /m"},{"name":"model.quality","verdict":"%s","detail":"%s is manifest-verified and loadable at /q"}]}}\n' "$verdict" "$quality"
    ;;
  *) exit 9 ;;
esac
SH
chmod +x "$work/tools/"* "$work/candidate"
export PATH="$work/tools:$PATH"
args=(--version v1.13.0 --quiet --dest "$work/dest" --native)
for FSFS_TEST_CASE in loader doctor wrong-model warn envelope failure loader-fail; do
  export FSFS_TEST_CASE
  printf '#!/bin/sh\necho "fsfs 1.12.0"\n' >"$work/dest/fsfs"
  chmod +x "$work/dest/fsfs"
  cp "$work/dest/fsfs" "$work/incumbent"
  : >"$work/calls"
  status=0
  bash "$installer" "${args[@]}" >"$work/output" 2>&1 || status=$?
  case "$FSFS_TEST_CASE" in
    loader|doctor)
      if [[ "$status" != 0 ]]; then cat "$work/output" >&2; exit 1; fi
      cmp "$work/dest/fsfs" "$work/candidate"
      for id in potion-multilingual-128m all-minilm-l6-v2-native; do
        grep -Fxq "download-models $id" "$work/calls"
        if [[ "$FSFS_TEST_CASE" == loader ]]; then
          grep -Fxq "download-models $id --verify-loader --offline" "$work/calls"
        elif [[ "$id" == all-minilm-l6-v2-native ]]; then
          grep -Fxq "download-models $id --verify --format json" "$work/calls"
        else
          grep -Fxq "download-models $id --verify" "$work/calls"
        fi
      done ;;
    *)
      [[ "$status" != 0 ]]
      cmp "$work/dest/fsfs" "$work/incumbent" ;;
  esac
  case "$FSFS_TEST_CASE" in
    loader*) if grep -q '^doctor ' "$work/calls"; then exit 1; fi ;;
    *) grep -q '^doctor --offline --format json --index-dir ' "$work/calls" ;;
  esac
done
export FSFS_TEST_CASE=loader FSFS_TEST_OS=Linux
bash "$installer" "${args[@]}"
cmp "$work/dest/fsfs" "$work/candidate"
for conflicting in '--lite' '--offline' '--verify' '--artifact-url archive' '--checksum 1234' '--checksum-url file'; do
  read -ra extra <<<"$conflicting"
  : >"$work/calls"
  if bash "$installer" "${args[@]}" "${extra[@]}" >"$work/output" 2>&1; then exit 1; fi
  [[ ! -s "$work/calls" ]]
  cmp "$work/dest/fsfs" "$work/candidate"
done
# A standard candidate that advertises loader admission receives its own pair;
# an older standard candidate keeps the established download + verify sequence.
export FSFS_TEST_STANDARD=1
for FSFS_TEST_CASE in loader doctor; do
  export FSFS_TEST_CASE
  : >"$work/calls"
  FSFS_INSTALL_CONTRACT_TEST=1 bash "$installer" provision "$work/candidate" >/dev/null
  if [[ "$FSFS_TEST_CASE" == loader ]]; then
    grep -Fxq "download-models all-minilm-l6-v2 --verify-loader --offline" "$work/calls"
  else
    grep -Fxq 'download-models all-minilm-l6-v2 --verify' "$work/calls"
    if grep -Eq -- '--verify-loader|^doctor ' "$work/calls"; then exit 1; fi
  fi
done
echo "Native installer source routing, staged loader checks, tagged doctor fallback, and incumbent preservation passed."
