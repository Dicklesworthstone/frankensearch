#!/usr/bin/env bash
# Executable quick-start gate (bd-fsfs-executable-quickstart-ci-ve3ul).
#
# Runs the public source-install quick start against the REAL binary and a tiny
# deterministic corpus, exactly as the README documents it — no hidden feature
# flags, no mocks. Fails closed on every regression class the 2026-08-01
# reality check found the hard way (docs/evidence/reality-check-20260801.md):
#
#   NONTERMINATION       `fsfs index` completes but never exits (exit 124 under cap)
#   MODEL-UNAVAILABLE    default build cannot index at all (exit 78)
#   MISSING-ARTIFACT     no sentinel/CURRENT/FSVI/Quill durability artifacts
#   INCOMPLETE-SENTINEL  sentinel exists but generation_complete != true
#   EMPTY-RESULTS        search returns no hits for a query with a known answer
#   WRONG-RANKING        the known-correct document is not rank 1
#   HASH-DEGRADATION     binary silently fell back to hash embedders
#   PROCESS-LEAK         child fsfs processes survive the run
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INDEX_DEADLINE_SECS=180
SKIP_BUILD=0
BINARY_OVERRIDE=""
NEGATIVE_PROBES=0
REQUIRE_SOURCE=0

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_executable_quickstart.sh [options]

Options:
  --binary PATH       Use an existing fsfs binary instead of building one.
  --skip-build        Alias for requiring --binary (fail if binary missing).
  --keep-artifacts    Explicit retention (artifacts are always retained).
  --negative-probes   Exercise the real failure cases and output-validator mutations.
  --require-source    Require an invocation-built binary from unchanged clean source.
  --index-deadline N  Seconds allowed for one bounded index run (default ${INDEX_DEADLINE_SECS}).
  --profile default|embedded
                      Which documented build to exercise (default: default).
  -h|--help           Show this help.

The gate builds the binary exactly as the README documents. Since the
loader-capable stock default landed (bd-fsfs-default-build-usable-6mtid),
the documented source path is the PLAIN default build — no feature flags —
with model artifacts provisioned at runtime; --profile embedded exercises
the zero-download release profile instead. Model inputs must already be
provisioned and SHA-verified via: scripts/rch-ensure-deps.sh --models-only
USAGE
}

BUILD_PROFILE="default"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY_OVERRIDE="${2:?}"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --keep-artifacts) shift ;;
    --negative-probes) NEGATIVE_PROBES=1; shift ;;
    --require-source) REQUIRE_SOURCE=1; shift ;;
    --index-deadline) INDEX_DEADLINE_SECS="${2:?}"; shift 2 ;;
    --profile) BUILD_PROFILE="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$INDEX_DEADLINE_SECS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid positive --index-deadline' >&2; exit 2; }

case "${BUILD_PROFILE}" in
  default|embedded) ;;
  *) echo "ERROR: invalid --profile '${BUILD_PROFILE}' (expected default|embedded)" >&2; exit 2 ;;
esac

fail() {
  local class="$1"; shift
  SHELL_FAILURE="$class"
  echo "" >&2
  echo "QUICKSTART GATE FAIL [${class}]: $*" >&2
  for log in "${WORK_DIR}"/index-run1.stderr "${WORK_DIR}"/index-run1.stdout \
             "${WORK_DIR}"/search.stderr "${WORK_DIR}"/search.json; do
    if [[ -s "${log}" ]]; then
      echo "---- tail ${log} ----" >&2
      tail -n 25 "${log}" >&2 || true
    fi
  done
  echo "Artifacts retained at: ${WORK_DIR}" >&2
  exit 1
}

cd "${ROOT_DIR}"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fsfs-quickstart.XXXXXXXX")"
# These are replay artifacts, including on failure. Never delete a run implicitly.
retain_artifacts() {
  local status=$?
  if [[ ! -f "$WORK_DIR/receipt.json" ]]; then
    python3 - "$WORK_DIR" "$status" "${SHELL_FAILURE:-SETUP}" <<'PY'
import json, sys
from pathlib import Path
work, status, reason = sys.argv[1:]
(Path(work) / 'receipt.json').write_text(json.dumps({
    'result': 'FAIL', 'stage': 'preflight-build-install',
    'exit': int(status), 'error': reason,
    'binary_source': 'unknown', 'artifacts': work,
}, indent=2) + '\n')
PY
  fi
  echo "Artifacts retained at: ${WORK_DIR}"
}
trap retain_artifacts EXIT

echo "[gate] work dir: ${WORK_DIR}"

# ── 0. Model provisioning preflight (explicit, SHA-verified, fail closed) ──
if ! scripts/rch-ensure-deps.sh --models-only --check >"${WORK_DIR}/models-check.log" 2>&1; then
  fail "MODEL-PROVISIONING" \
    "bundled model inputs absent or hash-mismatched; run scripts/rch-ensure-deps.sh --models-only first"
fi

# ── 1. Build the binary exactly as the README documents (default features) ──
CHECKER_REV="$(git -C "$ROOT_DIR" rev-parse HEAD)"
SOURCE_BEFORE="$(git -C "$ROOT_DIR" status --porcelain)"
BINARY_SOURCE="unknown"
BUILD_EXECUTABLE_SHA="unknown"
build_env=(env "HOME=$WORK_DIR/home" "XDG_CONFIG_HOME=$WORK_DIR/config"
  "XDG_CACHE_HOME=$WORK_DIR/cache" "XDG_DATA_HOME=$WORK_DIR/data"
  "CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}" "RUSTUP_HOME=${RUSTUP_HOME:-$HOME/.rustup}")
mkdir -p "$WORK_DIR/home" "$WORK_DIR/config" "$WORK_DIR/cache" "$WORK_DIR/data"
if [[ -n "${BINARY_OVERRIDE}" ]]; then
  FSFS_BIN="${BINARY_OVERRIDE}"
  [[ -x "${FSFS_BIN}" ]] || fail "SETUP" "--binary ${FSFS_BIN} is not executable"
  [[ "$REQUIRE_SOURCE" -eq 0 ]] || fail "BINARY-PROVENANCE" "--binary is smoke evidence; the checker cannot attest its source"
else
  [[ "${SKIP_BUILD}" -eq 1 ]] && fail "SETUP" "--skip-build requires --binary"
  if [[ "${BUILD_PROFILE}" == "embedded" ]]; then
    echo "[gate] building: cargo build --locked -p frankensearch-fsfs --bin fsfs --features embedded-models"
    "${build_env[@]}" cargo build --locked -p frankensearch-fsfs --bin fsfs --features embedded-models --message-format=json >"$WORK_DIR/build.jsonl" 2>"$WORK_DIR/build.stderr"
  else
    echo "[gate] building: cargo build --locked -p frankensearch-fsfs --bin fsfs (documented stock default, loader-capable)"
    "${build_env[@]}" cargo build --locked -p frankensearch-fsfs --bin fsfs --message-format=json >"$WORK_DIR/build.jsonl" 2>"$WORK_DIR/build.stderr"
  fi
  FSFS_BIN="$(python3 - "$WORK_DIR/build.jsonl" <<'PY'
import json, sys
artifacts = [json.loads(line) for line in open(sys.argv[1])]
paths = {a['executable'] for a in artifacts if a.get('reason') == 'compiler-artifact'
         and a.get('target', {}).get('name') == 'fsfs' and a.get('executable')}
assert len(paths) == 1, f'expected one Cargo fsfs executable, got {paths}'
assert any(a.get('reason') == 'build-finished' and a.get('success') for a in artifacts)
print(paths.pop())
PY
)"
  [[ -x "${FSFS_BIN}" ]] || fail "SETUP" "build produced no binary at ${FSFS_BIN}"
  BUILD_EXECUTABLE_SHA="$(sha256sum "$FSFS_BIN" | cut -d' ' -f1)"
  install_args=(--locked --debug --path crates/frankensearch-fsfs --root "$WORK_DIR/install"
    --target-dir "${CARGO_TARGET_DIR:-$ROOT_DIR/target}")
  [[ "$BUILD_PROFILE" == embedded ]] && install_args+=(--features embedded-models)
  echo '[gate] installing stock-feature debug executable into the fresh private Cargo root'
  "${build_env[@]}" cargo install "${install_args[@]}" >"$WORK_DIR/install.stdout" 2>"$WORK_DIR/install.stderr"
  FSFS_BIN="$WORK_DIR/install/bin/fsfs"
  [[ -x "$FSFS_BIN" && "$(sha256sum "$FSFS_BIN" | cut -d' ' -f1)" == "$BUILD_EXECUTABLE_SHA" ]] \
    || fail "BINARY-PROVENANCE" "Cargo install did not preserve the invocation-built executable"
  if [[ -z "$SOURCE_BEFORE" && -z "$(git status --porcelain)" && "$CHECKER_REV" == "$(git rev-parse HEAD)" ]]; then
    BINARY_SOURCE="$CHECKER_REV"
  fi
  [[ "$REQUIRE_SOURCE" -eq 0 || "$BINARY_SOURCE" != unknown ]] || fail "BINARY-PROVENANCE" "build source was dirty or changed during compilation"
fi

BIN_SHA256="$(sha256sum "${FSFS_BIN}" | cut -d' ' -f1)"

# ── 1b. Forbidden-runtime graph proof (AGENTS.md: asupersync only, no tokio) ──
echo "[gate] checker dependency graph: Tokio-family crates must be absent (${BUILD_PROFILE} profile)"
tree_args=(--locked -p frankensearch-fsfs --edges normal)
[[ "${BUILD_PROFILE}" == "embedded" ]] && tree_args+=(--features embedded-models)
"${build_env[@]}" cargo tree "${tree_args[@]}" \
  >"${WORK_DIR}/fsfs-dep-tree.txt" 2>"${WORK_DIR}/fsfs-dep-tree.err" \
  || fail "SETUP" "cargo tree failed for the checker source feature graph"
if grep -nE '(^|[^a-z-])(tokio|hyper|reqwest|axum|tower|async-std|smol) v[0-9]' \
    "${WORK_DIR}/fsfs-dep-tree.txt" >"${WORK_DIR}/forbidden-crates.txt"; then
  fail "FORBIDDEN-RUNTIME" \
    "Tokio-family crate(s) entered the checker source graph: $(head -3 "${WORK_DIR}/forbidden-crates.txt" | tr '\n' ' ')"
fi

# The executable and all negative cases share these validators. Mutated outputs
# are explicitly validator controls, never presented as real engine execution.
python3 - "$WORK_DIR" "$FSFS_BIN" "$CHECKER_REV" "$BINARY_SOURCE" \
  "$BIN_SHA256" "$INDEX_DEADLINE_SECS" "$NEGATIVE_PROBES" "$BUILD_PROFILE" "$BUILD_EXECUTABLE_SHA" <<'PY'
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import zlib

work, binary, checker, source, binary_sha, deadline, probes, profile, build_sha = sys.argv[1:]
work, binary = Path(work).resolve(), Path(binary).resolve()
deadline = int(deadline)
MAX_BYTES, MAX_LINES = 8 * 1024 * 1024, 10000
REDACTION_CANARY = 'fsfs-a5-private-token-7e93c1'
commands, controls = [], []
corpus, index = work / 'corpus', work / 'index'
model_root = Path(os.environ.get('FRANKENSEARCH_MODEL_DIR',
                  str(Path.home() / '.local/share/frankensearch/models'))).resolve()
# HOME/XDG isolation applies only to the child, never to the build toolchain.
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('FRANKENSEARCH_', 'FSFS_', 'HF_', 'HUGGINGFACE_'))}
for key, name in [('HOME', 'home'), ('XDG_CONFIG_HOME', 'config'),
                  ('XDG_CACHE_HOME', 'cache'), ('XDG_DATA_HOME', 'data')]:
    path = work / name
    path.mkdir(exist_ok=True)
    env[key] = str(path)
env.update(FRANKENSEARCH_MODEL_DIR=str(model_root), FRANKENSEARCH_OFFLINE='1',
           FRANKENSEARCH_ALLOW_DOWNLOAD='0', FRANKENSEARCH_CHECK_UPDATES='0',
           FRANKENSEARCH_LOG='info', FSFS_DISABLE_QUERY_CACHE='1', NO_COLOR='1',
           RAYON_NUM_THREADS='1', FSFS_A5_OWNER=str(work), OPENAI_API_KEY=REDACTION_CANARY)


class Refusal(Exception):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f'QUICKSTART GATE FAIL [{code}]: {detail}')


def require(ok, code, detail):
    if not ok:
        raise Refusal(code, detail)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def owned_processes():
    owner = f'FSFS_A5_OWNER={work}'.encode()
    found = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                if owner in (path / 'environ').read_bytes().split(b'\0'):
                    found.append(int(path.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    return found


def quiescent():
    remaining = owned_processes()
    require(not remaining, 'PROCESS-LEAK', f'owned child/listener PIDs still alive: {remaining}')


def redacted(data):
    require(REDACTION_CANARY.encode() not in data, 'LOG-SECRET', 'credential redaction canary escaped')


def run(label, args, *, cap=90, extra=None, expected=0, listener=None, trace_network=False, watch_probe=False):
    """Bound both output pipes and the whole owned process group, retaining exits."""
    output = [bytearray(), bytearray()]
    started = time.monotonic()
    argv = [str(binary), *map(str, args)]
    network_log = work / f'{label}.network'
    if trace_network:
        require(shutil.which('strace'), 'NETWORK-PROBE', 'strace is required on the Linux DSR host')
        argv = ['strace', '-f', '-qq', '-e', 'trace=network', '-o', str(network_log), *argv]
    p = subprocess.Popen(argv, cwd=work,
                         env=env | (extra or {}), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(p.stdout, selectors.EVENT_READ, 0)
    selector.register(p.stderr, selectors.EVENT_READ, 1)
    failure, rss, listener_checked, watch_ready = None, 0, False, False
    signals_sent = []
    try:
        while selector.get_map() or p.poll() is None:
            if watch_probe and not watch_ready and b'fsfs watch mode enabled; watcher started' in output[1]:
                watch_ready = True
                cap = time.monotonic() - started + 1
            if network_log.exists() and network_log.stat().st_size > MAX_BYTES:
                failure = 'LOG-LIMIT'
                break
            if listener is not None and listener.exists() and not listener_checked:
                # Same quiescence validator as every positive command, exercised
                # against a real listener that inherited this run's unique owner.
                reject('leaked-listener', 'PROCESS-LEAK', quiescent, 'real-executable')
                with socket.socket(socket.AF_UNIX) as client:
                    client.settimeout(2)
                    client.connect(str(listener))
                    client.sendall(b'quit\n')
                listener_checked = True
            if time.monotonic() - started >= cap:
                failure = 'WATCH-NOT-READY' if watch_probe and not watch_ready else 'NONTERMINATION'
                break
            try:
                status = Path(f'/proc/{p.pid}/status').read_text()
                values = re.findall(r'^Vm(?:HWM|RSS):\s+(\d+)', status, re.M)
                rss = max([rss, *map(int, values)])
            except FileNotFoundError:
                pass
            for key, _ in selector.select(0.02):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                remaining = MAX_BYTES - sum(c['bytes'] for c in commands) - sum(map(len, output))
                output[key.data].extend(chunk[:max(0, remaining)])
                if len(chunk) > remaining or sum(c['lines'] for c in commands) + sum(x.count(b'\n') for x in output) > MAX_LINES:
                    failure = 'LOG-LIMIT'
                    break
            if failure:
                break
        if failure:
            os.killpg(p.pid, signal.SIGTERM)
            signals_sent.append('SIGTERM')
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                signals_sent.append('SIGKILL')
        p.wait(timeout=5)
        try:
            os.killpg(p.pid, 0)
        except ProcessLookupError:
            pass
        else:
            failure = failure or 'PROCESS-LEAK'
            os.killpg(p.pid, signal.SIGKILL)
    finally:
        selector.close()
        p.stdout.close()
        p.stderr.close()
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=5)
        for suffix, data in zip(('stdout', 'stderr'), output):
            (work / f'{label}.{suffix}').write_bytes(data)
        commands.append(dict(label=label, argv=argv, pid=p.pid,
                             exit=p.returncode, elapsed_seconds=time.monotonic()-started,
                             signal=-p.returncode if p.returncode < 0 else None,
                             signals_sent=signals_sent, watch_ready=watch_ready,
                             refusal=failure, sampled_peak_rss_kib=rss,
                             rss_scope='sampled process leader; excludes descendants',
                             bytes=sum(map(len, output)), lines=sum(x.count(b'\n') for x in output)))
        print('[gate]', json.dumps(commands[-1]), flush=True)
    require(not failure, failure, label)
    for data in output:
        redacted(data)
    quiescent()
    require(listener is None or (listener_checked and not listener.exists()),
            'PROCESS-PROBE', 'listener must start, be detected, and be reclaimed')
    code = 'MODEL-UNAVAILABLE' if p.returncode == 78 else 'EXECUTION-ERROR'
    require(p.returncode == expected, code, f'{label}: exit={p.returncode}, expected={expected}')
    if trace_network:
        network = network_log.read_text()
        require('AF_INET' not in network, 'NETWORK-ACCESS', 'warm query attempted IPv4/IPv6 networking')
        commands[-1]['network_trace_sha256'] = sha(network_log)
        commands[-1]['inet_operations'] = 0
    return [x.decode('utf-8') for x in output]


def payload(raw):
    value = json.loads(raw)
    require(value.get('ok') is True, 'SEARCH-ERROR', 'unsuccessful envelope')
    return value['data']


def hits(data, *, hybrid=True):
    rows = data.get('hits') or []
    require(rows, 'EMPTY-RESULTS', 'known-answer query returned no result records')
    require(rows[0].get('path') == 'retry.md', 'WRONG-RANKING', rows[0].get('path'))
    require(rows[0].get('in_both_sources') is hybrid, 'WRONG-RANKING', 'source corroboration')
    require(rows[0].get('semantic_rank') == 0, 'WRONG-RANKING', 'semantic rank')
    if not hybrid:
        require(all(x.get('lexical_rank') is None for x in rows), 'WRONG-RANKING', 'vector-only has lexical rank')
    require(not (data.get('index_freshness') or {}).get('degraded'), 'HASH-DEGRADATION', 'degraded search')


def fsvi(path, quality=False):
    require(path.is_file() and path.stat().st_size > 0,
            'MISSING-QUALITY' if quality else 'MISSING-VECTOR', str(path))
    data = path.read_bytes()
    require(data[:6] == b'FSVI\x01\x00', 'VECTOR-FORMAT', 'expected current v1 FSVI; update at format cutover')
    length = struct.unpack_from('<H', data, 6)[0]
    name = data[8:8+length].decode()
    offset = 8 + length
    rev_len = struct.unpack_from('<H', data, offset)[0]
    revision = data[offset+2:offset+2+rev_len].decode()
    offset += 2 + rev_len
    dimension, quantization, count, slab = struct.unpack_from('<IB3xQQ', data, offset)
    crc_offset = offset + 24
    require(zlib.crc32(data[:crc_offset]) == struct.unpack_from('<I', data, crc_offset)[0],
            'VECTOR-FORMAT', 'header checksum mismatch')
    require(not any(x in name.lower() for x in ('hash', 'fnv')), 'HASH-DEGRADATION', name)
    require(count == 3 and dimension > 0 and slab < len(data), 'VECTOR-FORMAT', 'shape or missing vector slab')
    require('minilm' in name.lower() if quality else ('potion' in name.lower() or 'model2vec' in name.lower()),
            'HASH-DEGRADATION', f'unexpected producer {name}')
    require(not quality or dimension == 384, 'VECTOR-FORMAT', 'MiniLM dimension')
    return dict(path=str(path), sha256=sha(path), producer=name, revision=revision,
                dimension=dimension, records=count, quantization=quantization)


def artifacts(root):
    sentinel = root / 'index_sentinel.json'
    require(sentinel.is_file(), 'MISSING-SENTINEL', str(sentinel))
    state = json.loads(sentinel.read_text())
    require(state.get('generation_complete') is True and state.get('indexed_files') == 3,
            'INCOMPLETE-SENTINEL', state)
    lexical = root / 'lexical'
    require(lexical.is_dir(), 'MISSING-LEXICAL', str(lexical))
    current = lexical / 'CURRENT'
    require(current.is_file(), 'MISSING-CURRENT', str(current))
    pointer = current.read_bytes()
    require(len(pointer) >= 23 and pointer[:8] == b'FSLXCUR\0', 'MISSING-CURRENT', 'invalid pointer header')
    require(zlib.crc32(pointer[:-4]) == struct.unpack_from('<I', pointer, len(pointer)-4)[0],
            'MISSING-CURRENT', 'pointer checksum')
    version, engine, length = struct.unpack_from('<IBH', pointer, 8)
    require(version == 1 and engine == 1 and len(pointer) == length + 23,
            'MISSING-CURRENT', 'pointer version, engine or length')
    target = pointer[15:15+length].decode()
    require(target and Path(target).name == target and target not in ('.', '..') and '\\' not in target and '\0' not in target,
            'MISSING-CURRENT', 'CURRENT must name one local engine directory')
    manifest = lexical / target / 'MANIFEST'
    require(manifest.is_file() and manifest.stat().st_size > 0, 'MISSING-MANIFEST', str(manifest))
    fast, quality = fsvi(root / 'vector/index.fsvi'), fsvi(root / 'vector/quality.fsvi', True)
    require(fast['producer'] != quality['producer'] and fast['sha256'] != quality['sha256'],
            'MISSING-QUALITY', 'quality must be independently produced')
    return dict(sentinel=state, current=target, manifest_sha256=sha(manifest), fast=fast, quality=quality)


def phases(raw):
    frames = [json.loads(line) for line in raw.splitlines()]
    require(len(frames) >= 4 and frames[0].get('event') == 'started', 'MISSING-PHASE', 'stream start')
    require(frames[0].get('payload', {}).get('semantic_admitted') is True, 'HASH-DEGRADATION', 'stream producer')
    require([f.get('seq') for f in frames] == list(range(len(frames)))
            and len({f.get('stream_id') for f in frames}) == 1, 'MISSING-PHASE', 'stream identity or sequence')
    codes = [x.get('payload', {}).get('reason_code') for x in frames]
    required = ['query.stream.initial_ready', 'query.stream.refined_ready']
    positions = []
    for code in required:
        require(codes.count(code) == 1, 'MISSING-PHASE', f'{code}: {codes}')
        positions.append(codes.index(code))
    require(positions == sorted(positions), 'MISSING-PHASE', 'phase order')
    terminals = [i for i, frame in enumerate(frames) if frame.get('event') == 'terminal']
    require(terminals == [len(frames)-1], 'MISSING-PHASE', 'exactly one final terminal')
    terminal = frames[-1]['payload']
    require(terminal.get('status') == 'completed' and terminal.get('exit_code') == 0, 'MISSING-PHASE', terminal)
    for begin, end in zip(positions, [positions[1], len(frames)-1]):
        rows = [frame['payload']['item'] for frame in frames[begin+1:end] if frame.get('event') == 'result']
        require(len(rows) == frames[begin]['payload']['completed_units'], 'MISSING-PHASE', 'announced/result count mismatch')
        hits({'hits': rows})
    return frames


def provenance(value):
    require(value['binary_source'] != 'unknown' and value['origin'] == 'invocation-build',
            'BINARY-PROVENANCE', 'checker revision cannot attest an override binary')


def reject(name, expected, action, kind):
    try:
        action()
    except Refusal as error:
        require(error.code == expected, 'WRONG-REFUSAL', f'{name}: {error}')
        controls.append(dict(name=name, expected=expected, observed=error.code, detail=str(error), kind=kind))
        print('[gate] RED', json.dumps(controls[-1]), flush=True)
    else:
        raise Refusal('NEGATIVE-ACCEPTED', name)


receipt = dict(checker_revision=checker, binary_source=source, binary_sha256=binary_sha,
               build_executable_sha256=build_sha, executed_path=str(binary),
               origin='supplied-binary' if source == 'unknown' else 'invocation-build',
               profile=profile, model_root=str(model_root), commands=commands, controls=controls,
               result='FAIL', network_policy='offline; warm query network syscalls traced separately')
try:
    require(sys.platform.startswith('linux'), 'SETUP', 'A5 uses Linux /proc and pidfd process ownership')
    corpus.mkdir()
    for name, text in [('retry.md', 'Retry backoff uses exponential delay with jitter to avoid thundering herds.'),
                       ('parser.md', 'The parser tokenizes UTF-8 input into normalized terms.'),
                       ('concurrency.md', 'Structured concurrency ensures no orphan tasks survive scope exit.')]:
        (corpus / name).write_text(text + '\n')
    for ordinal in (1, 2):
        raw, log = run(f'index-run{ordinal}', ['index', corpus, '--index-dir', index, '--format', 'json'], cap=deadline)
        payload(raw)
        receipt[f'generation{ordinal}'] = artifacts(index)
    # Actual loader output and the authoritative on-disk receipts are retained;
    # status alone is never accepted as proof that inference happened.
    require('Model2Vec model loaded' in log, 'MODEL-IDENTITY', 'fast loader was not observed')
    receipts = {}
    for directory in ('potion-multilingual-128M', 'all-MiniLM-L6-v2'):
        receipt_path = model_root / directory / '.verified'
        require(receipt_path.is_file(), 'MODEL-IDENTITY', f'no verification receipt for {directory}')
        receipts[directory] = json.loads(receipt_path.read_text())
        fingerprint = receipts[directory]['manifest_fingerprint']
        require(any('model verification receipt accepted' in line
                    and f'receipt_manifest_fingerprint={fingerprint}' in line for line in log.splitlines()),
                'MODEL-IDENTITY', f'loader did not admit the recorded {directory} receipt')
    receipt['model_receipts'] = receipts
    for tier in ('fast', 'quality'):
        actual_identity = receipt['generation2'][tier]['revision']
        require(any('model loaded' in line and f'identity={actual_identity}' in line for line in log.splitlines()),
                'MODEL-IDENTITY', f'{tier} FSVI must bind the actually loaded producer')
    query = 'how does retry backoff work'
    search_args = ['search', query, '--index-dir', index, '--limit', '3', '--no-daemon']
    raw, _ = run('hybrid', [*search_args, '--format', 'json'])
    result = payload(raw)
    hits(result)
    require(result.get('phase') == 'refined', 'MISSING-PHASE', result.get('phase'))
    raw, _ = run('warm-offline', [*search_args, '--format', 'json'], trace_network=True)
    hits(payload(raw))
    raw, _ = run('progressive', [*search_args, '--stream', '--format', 'jsonl'])
    stream = phases(raw)
    raw, _ = run('status', ['status', '--index-dir', index, '--no-watch-mode', '--format', 'json'])
    status = payload(raw)
    tiers = {x.get('tier'): x.get('name', '') for x in status.get('models', [])}
    require(all(tiers.get(t) and 'hash' not in tiers[t].lower() for t in ('fast', 'quality')),
            'HASH-DEGRADATION', tiers)
    # A real vector-only generation exercises the intended typed degraded lane.
    semantic = work / 'semantic-only'
    (semantic / 'vector').mkdir(parents=True)
    shutil.copy2(index / 'vector/index.fsvi', semantic / 'vector/index.fsvi')
    raw, _ = run('semantic-only', ['search', query, '--index-dir', semantic, '--no-daemon', '--format', 'json'])
    semantic_result = payload(raw)
    hits(semantic_result, hybrid=False)
    require(semantic_result.get('phase') == 'initial', 'MISSING-PHASE', 'quality-less search must stay Initial')
    if probes == '1':
        # Remove authority only from new copies. The live positive generation
        # and shared model cache remain untouched; absent inputs are not mocks.
        omitted = [('missing-current', 'lexical/CURRENT', 'MISSING-CURRENT'),
                   ('missing-sentinel', 'index_sentinel.json', 'MISSING-SENTINEL'),
                   ('missing-manifest', 'lexical/' + receipt['generation2']['current'] + '/MANIFEST', 'MISSING-MANIFEST'),
                   ('missing-vector', 'vector/index.fsvi', 'MISSING-VECTOR'),
                   ('missing-lexical', 'lexical', 'MISSING-LEXICAL'),
                   ('missing-quality', 'vector/quality.fsvi', 'MISSING-QUALITY')]
        for name, absent, expected in omitted:
            target = work / name
            def ignore(directory, names):
                return [n for n in names if str((Path(directory) / n).relative_to(index)) == absent]
            shutil.copytree(index, target, ignore=ignore)
            reject(name, expected, lambda: artifacts(target), 'mutated-real-artifact')
        incomplete = work / 'incomplete-sentinel'
        shutil.copytree(index, incomplete)
        state = json.loads((incomplete/'index_sentinel.json').read_text())
        state['generation_complete'] = False
        (incomplete/'index_sentinel.json').write_text(json.dumps(state))
        reject('incomplete-sentinel', 'INCOMPLETE-SENTINEL', lambda: artifacts(incomplete), 'mutated-real-artifact')
        bad = copy.deepcopy(result)
        bad['hits'] = []
        reject('empty-result', 'EMPTY-RESULTS', lambda: hits(bad), 'output-validator-mutation')
        bad = copy.deepcopy(result)
        bad['hits'][0]['path'] = 'parser.md'
        reject('wrong-result', 'WRONG-RANKING', lambda: hits(bad), 'output-validator-mutation')
        hash_path = work / 'hash-masquerade.fsvi'
        data = bytearray((index/'vector/index.fsvi').read_bytes())
        name_len = struct.unpack_from('<H', data, 6)[0]
        data[8:8+name_len] = b'hash-control'.ljust(name_len)
        rev_len = struct.unpack_from('<H', data, 8+name_len)[0]
        crc_offset = 8 + name_len + 2 + rev_len + 24
        struct.pack_into('<I', data, crc_offset, zlib.crc32(data[:crc_offset]))
        hash_path.write_bytes(data)
        reject('hash-only-masquerade', 'HASH-DEGRADATION', lambda: fsvi(hash_path), 'mutated-real-artifact')
        bad_stream = [f for f in stream if f.get('payload', {}).get('reason_code') != 'query.stream.refined_ready']
        def encode_frames(frames):
            return '\n'.join(json.dumps(dict(frame, seq=i)) for i, frame in enumerate(frames))
        reject('missing-refined', 'MISSING-PHASE', lambda: phases(encode_frames(bad_stream)), 'output-validator-mutation')
        empty_phases = [f for f in stream if f.get('event') != 'result']
        reject('empty-phase-results', 'MISSING-PHASE', lambda: phases(encode_frames(empty_phases)), 'output-validator-mutation')
        false_source = dict(receipt, binary_source=checker, origin='supplied-binary')
        reject('false-binary-provenance', 'BINARY-PROVENANCE', lambda: provenance(false_source), 'provenance-validator-mutation')
        reject('redaction-canary', 'LOG-SECRET', lambda: redacted(REDACTION_CANARY.encode()), 'log-validator-mutation')
        empty_models = work / 'empty-models'
        empty_models.mkdir()
        reject('unavailable-model', 'MODEL-UNAVAILABLE',
               lambda: run('missing-model', ['index', corpus, '--index-dir', work/'missing-model-index', '--format', 'json'],
                           cap=30, extra={'FRANKENSEARCH_MODEL_DIR': str(empty_models)}), 'real-executable')
        missing = json.loads((work / 'missing-model.stdout').read_text())
        require(missing.get('error', {}).get('code') == 'embedder_unavailable', 'WRONG-REFUSAL', missing)
        corrupt_models = work / 'corrupt-models'
        corrupt = corrupt_models / 'potion-multilingual-128M'
        corrupt.mkdir(parents=True)
        for entry in (model_root / 'potion-multilingual-128M').iterdir():
            if entry.is_file() and entry.name != '.verified':
                shutil.copy2(entry, corrupt / entry.name)
        tokenizer = corrupt / 'tokenizer.json'
        data = tokenizer.read_bytes()
        tokenizer.write_bytes(b'!' + data[1:])
        raw, _ = run('corrupt-verify', ['download-models', 'potion-multilingual-128m', '--verify', '--format', 'json'],
                     cap=30, expected=1, extra={'FRANKENSEARCH_MODEL_DIR': str(corrupt_models)})
        require(json.loads(raw).get('error', {}).get('code') == 'hash_mismatch', 'WRONG-REFUSAL', raw)
        reject('corrupt-model', 'MODEL-UNAVAILABLE',
               lambda: run('corrupt-model', ['index', corpus, '--index-dir', work/'corrupt-model-index', '--format', 'json'],
                           cap=30, extra={'FRANKENSEARCH_MODEL_DIR': str(corrupt_models)}), 'real-executable')
        require('model verification receipt accepted' not in (work/'corrupt-model.stderr').read_text(),
                'MODEL-IDENTITY', 'corrupt cache must not be admitted by a verification receipt')
        # Restore the original one-shot-vs-watch failure using the public watch
        # command on a private copy, not a modified binary or a sleeping stub.
        reject('nontermination', 'NONTERMINATION',
               lambda: run('watch-deadline', ['watch', corpus, '--index-dir', work/'watch-index', '--format', 'jsonl'], cap=deadline, watch_probe=True),
               'real-executable')
        # A live daemon listener must be detected independently of parent exit.
        endpoint = Path('/tmp') / f'fsfs-a5-{os.getpid()}.sock'
        run('leaked-listener', ['serve', '--daemon', '--daemon-socket', endpoint,
                               '--index-dir', index], cap=60, listener=endpoint)
        require(len(controls) == 18, 'NEGATIVE-INVENTORY', len(controls))
    require(sha(binary) == binary_sha, 'BINARY-PROVENANCE', 'executable changed during run')
    receipt['result'] = 'PASS'
except (Refusal, OSError, ValueError, KeyError, struct.error) as error:
    receipt['error'] = str(error)
    print(str(error), file=sys.stderr)
    sys.exit(1)
finally:
    # Reclaim only processes carrying this invocation's unguessable directory
    # identity. pidfds keep a racing PID reuse from targeting a foreign process.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in owned_processes():
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                if pid in owned_processes():
                    signal.pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)
        until = time.monotonic() + 5
        while owned_processes() and time.monotonic() < until:
            time.sleep(0.05)
    receipt['remaining_owned_processes'] = owned_processes()
    if receipt['remaining_owned_processes']:
        receipt['result'] = 'FAIL'
        receipt['error'] = 'PROCESS-LEAK: owned processes could not be reclaimed'
    (work / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
require(receipt['result'] == 'PASS', 'PROCESS-LEAK', receipt.get('error'))
print(f'[gate] PASS binary_source={source} checker_revision={checker} binary_sha256={binary_sha}')
print('Result: PASS')
PY
