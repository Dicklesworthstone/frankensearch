#!/usr/bin/env python3
"""Run Quill native witness and replay regressions; retain the full gauntlet route.

Invoked by quality-gate.sh on the admitted validation host. This is a test
driver, not a conformance or performance certificate. --full runs both Cargo
feature configurations, including the slow evidence-assembly tests. Existing
ignored tests retain their separate nightly/isolated/review-only obligations.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "crates/frankensearch-quill-gauntlet"
NATIVE = "native_enriched_witness_live"
LIBRARY = "frankensearch_quill_gauntlet"
ORACLE = "the_committed_expectations_hold_against_real_tantivy"
QUILL = "the_committed_expectations_hold_against_real_quill"
WITNESS = "native_enriched_witness::tests::"
REPLAY = "engine::tests::typed_query_"
MUTATION = WITNESS + "a_common_mode_mutation_passes_agreement_and_still_fails_the_oracle"
# Unchanged native integration: 2.915s; all 39 witness unit cases: 0.102s
# summed test time on vmi1152480, 2026-09-09. Replay regressions join this lane
# after the full run exposed the seed and atomic-publication bugs. The entire
# engine module took 10.9s summed test time in that run. Allow host headroom without
# including Cargo compilation in the runtime budget. No test workload changes.
BOUNDED_SECONDS = 90


class Refusal(Exception):
    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code = code


def emit(event, **fields):
    print(json.dumps({"quill_gate": event, **fields}), flush=True)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def invoke(argv, seconds, label, logs):
    """Keep both output streams and reap this invocation's entire process group."""
    started = time.monotonic()
    emit("started", label=label, argv=argv, budget_seconds=seconds)
    stdout_path = logs / f"{label}.stdout"
    stderr_path = logs / f"{label}.stderr"
    timed_out = False
    # Retain child output even if the remote controller interrupts this driver.
    # Report only observed output growth, not a synthetic liveness heartbeat.
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file, \
            selectors.DefaultSelector() as streams:
        process = subprocess.Popen(
            argv, cwd=PACKAGE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        streams.register(process.stdout, selectors.EVENT_READ, stdout_file)
        streams.register(process.stderr, selectors.EVENT_READ, stderr_file)
        previous_sizes = (0, 0)
        last_output_time = started

        def collect(deadline):
            nonlocal previous_sizes, last_output_time
            while streams.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, seconds)
                for key, _ in streams.select(timeout=min(5, remaining)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if chunk:
                        key.data.write(chunk)
                        key.data.flush()
                    else:
                        streams.unregister(key.fileobj)
                        key.fileobj.close()
                sizes = (stdout_file.tell(), stderr_file.tell())
                now = time.monotonic()
                if sizes != previous_sizes and (now - last_output_time >= 5 or not streams.get_map()):
                    emit("output", label=label, stdout_bytes=sizes[0], stderr_bytes=sizes[1])
                    previous_sizes = sizes
                    last_output_time = now
            process.wait(timeout=max(deadline - time.monotonic(), 0.0001))

        try:
            collect(started + max(seconds, 0.0001))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                collect(time.monotonic() + 5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                collect(float("inf"))
                process.wait()
        except BaseException:
            # A log write failure must not leave the command running unobserved.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        finally:
            for key in list(streams.get_map().values()):
                key.fileobj.close()
    stdout = stdout_path.read_text()
    stderr = stderr_path.read_text()
    if stderr:
        print(stderr, end="", file=sys.stderr, flush=True)
    emit("finished", label=label, exit_code=process.returncode,
         timed_out=timed_out, elapsed_seconds=time.monotonic() - started,
         stdout=str(logs / f"{label}.stdout"), stderr=str(logs / f"{label}.stderr"))
    if timed_out:
        raise Refusal("TIMEOUT", label)
    return process.returncode, stdout


def events(stdout):
    try:
        return [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except ValueError as error:
        raise Refusal("INVALID_OUTPUT", str(error)) from error


def build(configuration, logs):
    argv = ["cargo", "test", "--locked", "-p", "frankensearch-quill-gauntlet",
            "--no-run", "--message-format=json", "--tests"]
    if configuration == "all":
        argv.append("--all-features")
    else:
        argv.append("--no-default-features")
    code, output = invoke(argv, 7200, f"build-{configuration}", logs)
    rows = events(output)
    if code or not any(row.get("reason") == "build-finished" and row.get("success")
                       for row in rows):
        raise Refusal("BUILD_FAILED", configuration)
    binaries = {}
    for row in rows:
        if (row.get("reason") == "compiler-artifact" and row.get("executable")
                and row["profile"]["test"]
                and Path(row["manifest_path"]).parent == PACKAGE):
            name = row["target"]["name"].replace("-", "_")
            if name in binaries:
                raise Refusal("DUPLICATE_BINARY", name)
            binaries[name] = Path(row["executable"])
    if NATIVE not in binaries or LIBRARY not in binaries:
        raise Refusal("MISSING_BINARY", configuration)
    return binaries


def inventory(binary, label, logs):
    code, output = invoke([str(binary), "--list", "--format=json", "-Z", "unstable-options"],
                          30, f"list-{label}", logs)
    rows = events(output)
    tests = [row for row in rows if row.get("type") == "test"]
    terminals = [row for row in rows if row.get("event") == "completed"]
    if (code or len(terminals) != 1 or terminals[0].get("tests") != len(tests)
            or len({row["name"] for row in tests}) != len(tests)):
        raise Refusal("INVALID_INVENTORY", label)
    emit("inventory", label=label, binary=str(binary), sha256=digest(binary), tests=tests)
    return tests


def require_oracle(tests):
    runnable = {row["name"] for row in tests if not row["ignore"]}
    if not {ORACLE, QUILL} <= runnable:
        raise Refusal("MISSING_ORACLE", f"required runnable tests: {ORACLE}, {QUILL}")


def validate(output, code, selected):
    if not selected:
        raise Refusal("ZERO_TESTS", "selection is empty")
    rows = events(output)
    terminals = [row for row in rows if row.get("type") == "suite"
                 and row.get("event") in {"ok", "failed"}]
    completed = [row for row in rows if row.get("type") == "test"
                 and row.get("event") in {"ok", "failed", "ignored"}]
    if len(terminals) != 1:
        raise Refusal("MISSING_TERMINAL", f"got {len(terminals)} suite terminals")
    names = [row["name"] for row in completed]
    if len(names) != len(set(names)) or set(names) != set(selected):
        raise Refusal("INVENTORY_MISMATCH", "executed names differ from Cargo inventory")
    terminal = terminals[0]
    counts = {event: sum(row["event"] == event for row in completed)
              for event in ("ok", "failed", "ignored")}
    if any(terminal.get(key) != counts[event]
           for key, event in (("passed", "ok"), ("failed", "failed"), ("ignored", "ignored"))):
        raise Refusal("COUNT_MISMATCH", str(terminal))
    emit("counts", passed=counts["ok"], failed=counts["failed"], ignored=counts["ignored"])
    if code or counts["failed"] or terminal["event"] != "ok":
        raise Refusal("TEST_FAILED", str(terminal))
    if not counts["ok"]:
        raise Refusal("ZERO_TESTS", "no tests passed")


def execute(binary, tests, label, logs, seconds, selector=None):
    selected = [row["name"] for row in tests if selector is None or selector in row["name"]]
    before = digest(binary)
    argv = [str(binary), "--format=json", "-Z", "unstable-options",
            "--report-time", "--test-threads=2"]
    if selector is not None:
        argv.append(selector)
    code, output = invoke(argv, seconds, label, logs)
    if digest(binary) != before:
        raise Refusal("BINARY_CHANGED", str(binary))
    validate(output, code, selected)
    return output


def expect_refusal(code, operation):
    try:
        operation()
    except Refusal as error:
        if error.code != code:
            raise
        emit("negative_control", expected=code, observed=str(error))
    else:
        raise Refusal("NEGATIVE_ACCEPTED", code)


def probe_output_retention(logs):
    # The real child refuses success unless both streams are readable before it
    # exits. The former communicate-then-write implementation fails this probe.
    child = """import pathlib, sys, time
print('live-out', flush=True)
print('live-err', file=sys.stderr, flush=True)
paths = [pathlib.Path(p) for p in sys.argv[1:]]
deadline = time.monotonic() + 2
while time.monotonic() < deadline:
    if all(p.exists() and marker in p.read_text() for p, marker in zip(paths, ['live-out', 'live-err'])):
        sys.exit(0)
    time.sleep(0.01)
sys.exit(9)
"""
    label = "probe-live-output"
    code, output = invoke([sys.executable, "-c", child,
                           str(logs / f"{label}.stdout"), str(logs / f"{label}.stderr")],
                          10, label, logs)
    if code or output != "live-out\n" or (logs / f"{label}.stderr").read_text() != "live-err\n":
        raise Refusal("OUTPUT_RETENTION_FAILED", label)
    emit("positive_control", observed="both streams retained before child exit")
    # Parent exit must not silently truncate output inherited by a descendant;
    # the unchanged timeout still kills the invocation's whole process group.
    descendant = "import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])"
    expect_refusal("TIMEOUT", lambda: invoke([sys.executable, "-c", descendant],
                                           0.5, "probe-inherited-output-timeout", logs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="default + all-feature complete nonignored gauntlet")
    mode.add_argument("--probes", action="store_true", help="bounded lane plus actual zero/oracle/timeout/output negatives")
    args = parser.parse_args()
    # Cargo-configuration security fixtures create repositories under TMPDIR.
    # Keeping those below this checkout accidentally gives them an unbound
    # ancestor .cargo/config.toml. A private canonical system-temp directory
    # also keeps the replay capability's no-symlink walk meaningful.
    logs = Path(tempfile.mkdtemp(prefix="quill-gate-", dir=Path("/tmp").resolve()))
    os.environ["TMPDIR"] = str(logs)
    os.environ.pop("RUST_TEST_NOCAPTURE", None)
    emit("source", head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip(),
         lock_sha256=digest(ROOT / "Cargo.lock"), host=os.uname().nodename, logs=str(logs))
    if args.probes:
        probe_output_retention(logs)
    failures = []
    for configuration in (["default", "all"] if args.full else ["all"]):
        binaries = build(configuration, logs)
        listed = {name: inventory(path, f"{configuration}-{name}", logs)
                  for name, path in sorted(binaries.items())}
        if configuration == "all":
            require_oracle(listed[NATIVE])
            mutation = [row for row in listed[LIBRARY] if row["name"] == MUTATION and not row["ignore"]]
            if not mutation:
                raise Refusal("MISSING_MUTATION", MUTATION)
        deadline = time.monotonic() + (28800 if args.full else BOUNDED_SECONDS)
        for name, tests in listed.items():
            bounded = name in {NATIVE, LIBRARY}
            emit("route", configuration=configuration, binary=name,
                 bounded=[row["name"] for row in tests if configuration == "all" and bounded
                          and (name == NATIVE or row["name"].startswith((WITNESS, REPLAY)))],
                 full=[row["name"] for row in tests if not row["ignore"]],
                 separate=[row for row in tests if row["ignore"]])
            if not tests:
                emit("empty_target", binary=name, counted_as_pass=False)
                continue
            if not args.full and not bounded:
                continue
            output = None
            selectors = [None] if args.full or name == NATIVE else [WITNESS, REPLAY]
            for ordinal, selector in enumerate(selectors):
                selected = [row for row in tests if selector is None or selector in row["name"]]
                if any(row["ignore"] for row in selected) and not args.full:
                    raise Refusal("REQUIRED_TEST_IGNORED", name)
                try:
                    output = execute(binaries[name], tests, f"run-{configuration}-{name}-{ordinal}", logs,
                                     deadline - time.monotonic(), selector)
                except Refusal as error:
                    emit("refused", reason=str(error))
                    failures.append(str(error))
                    continue
            if args.probes and name == NATIVE and output is not None:
                expect_refusal("ZERO_TESTS", lambda: execute(binaries[name], tests, "probe-zero", logs,
                               30, "__quill_gate_deliberately_absent_test__"))
                expect_refusal("TIMEOUT", lambda: execute(binaries[name], tests, "probe-timeout", logs, 0.0001))
                without_terminal = "\n".join(json.dumps(row) for row in events(output)
                                             if not (row.get("type") == "suite" and row.get("event") == "ok"))
                expect_refusal("MISSING_TERMINAL", lambda: validate(without_terminal, 0, [row["name"] for row in tests]))
        if args.probes:
            default_binaries = build("default", logs)
            default_inventory = {}
            for name, binary in sorted(default_binaries.items()):
                tests = inventory(binary, f"default-{name}", logs)
                default_inventory[name] = tests
                emit("route", configuration="default", binary=name, bounded=[],
                     full=[row["name"] for row in tests if not row["ignore"]],
                     separate=[row for row in tests if row["ignore"]])
            expect_refusal("MISSING_ORACLE", lambda: require_oracle(default_inventory[NATIVE]))
    if failures:
        raise Refusal("LANE_FAILED", "; ".join(failures))
    emit("passed", mode="full" if args.full else "probes" if args.probes else "bounded",
         scope="Native witness and typed-query replay correctness; no full conformance or performance claim")


if __name__ == "__main__":
    try:
        main()
    except (Refusal, OSError, subprocess.SubprocessError) as error:
        emit("refused", reason=str(error))
        sys.exit(1)
