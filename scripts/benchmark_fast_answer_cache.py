#!/usr/bin/env python3
r"""Diagnostic maintenance measurement for issue #73, using a real paired A/B run.

Requires Python 3.11+, the pinned Potion artifacts, and two successfully built
binaries. This script never builds fsfs or substitutes simulated measurements.
Its output is diagnostic maintenance evidence, not promotion/release evidence or
a competitive win. The recorded feature/profile selection defines its scope;
in particular, a native-only build is not proof for the default build.
The source roots must be clean Git worktrees, including non-ignored untracked
files; put build outputs elsewhere. Both HEAD and tracked-diff hashes are checked
before and after the run. A clean diff's SHA-256 is
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.

Example:
    python3 scripts/benchmark_fast_answer_cache.py \
        --baseline /work/bin/fsfs-before --baseline-source /work/before \
        --candidate /work/bin/fsfs-after --candidate-source /work/after \
        --models /work/models --build-metadata /work/builds.json \
        --output /work/evidence/cache-ab --rounds 5

The output directory must not exist and must be outside both source worktrees
and the input model root. All paths may be relative; they are resolved before
changing child working directories. Only the two pinned Potion files are copied
from --models/potion-multilingual-128M. The input files and any input receipt are
read-only. Both executables are also copied and hash-checked. The baseline's
offline download-models --verify command creates the private model receipt before
indexing and timing. With --receipt-mode valid (the default), warm searches must
log receipt acceptance. The absent and malformed modes preserve that valid receipt
outside the model directory, then remove it by renaming or replace it with invalid
JSON in the private installation. These modes must not mint or repair a receipt
during indexing or search. No input model files or receipts are modified.

--build-metadata describes the actual successful builds, using the SAME command,
relevant effective environment, toolchain, profile, and features for both arms:
    {
      "schema_version": 1,
      "command": ["cargo", "build", "--locked", "--release",
                  "-p", "frankensearch-fsfs", "--bin", "fsfs"],
      "environment": {"CARGO_PROFILE_RELEASE_LTO": "false",
                      "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16"},
      "feature_profile": {"cargo_profile": "release", "default_features": true,
                          "all_features": false, "features": []},
      "rustc_vv": "the complete actual rustc -Vv output",
      "arms": {
        "baseline": {"source_head": "40-character commit SHA",
                     "tracked_diff_sha256": "64-character SHA-256",
                     "binary_sha256": "64-character SHA-256",
                     "build_exit_code": 0},
        "candidate": {"source_head": "40-character commit SHA",
                      "tracked_diff_sha256": "64-character SHA-256",
                      "binary_sha256": "64-character SHA-256",
                      "build_exit_code": 0}
      }
    }

Use a direct cargo build argv from each source root (an optional +toolchain
argument is supported). Spell out profile/feature flags or use separate short
options; combined short options such as -qr are rejected.
feature_profile must match the command's profile and feature flags exactly;
features is a sorted list of requested feature names. Record all build-affecting
overrides, including RUSTFLAGS and ORT linkage when used. Do not fill this file
with planned commands after an unsuccessful build.
The runner verifies the supplied evidence's source and binary bindings; it cannot
reconstruct an unrecorded build.

The default --layout legacy measures the original regression, explicitly setting
FSFS_COMPLETE_GENERATIONS=0. --layout complete sets FSFS_COMPLETE_GENERATIONS=1
and measures the store-owned answer cache for immutable complete generations.
Use a complete-generation baseline before that cache was added (for example
448af4c): it must leave no persisted answer after either request. The candidate
must write exactly one record under the store's query_cache directory, bound to
the selected generation ID and the actual FSFS-BUNDLE.json digest. The selected
pointer and every sealed bundle file must remain unchanged after each request.
Cold/warm here means the persisted ANSWER cache, not the OS page cache. Each arm
and round receives a copy of one index in the selected layout. The baseline must
load Model2Vec once on both requests; the candidate must load once cold and zero
times warm. Payloads, ordered hits/scores, and cache bytes, mtime, and record count
are checked. Both binaries must support the selected layout.
The default --scenario semantic-hits measures the original positive-answer
regression. --scenario filtered-empty instead uses an excluding extension filter
and verifies an unfiltered semantic positive control for EACH arm on separate
indexes before timing. With legacy layout, use a baseline before e700da7 (for
example 6e6d771) to isolate that zero-result correction: its warm request recomputes
and rewrites the answer, while the candidate must replay it without loading
weights or rewriting the cache. Complete filtered-empty payloads must agree
across arms and rounds. Both arms must retain identical cold/warm answer/cache
bytes; only that explicit legacy scenario requires the baseline's cache mtime to
change. The original scenario's unchanged-mtime gate remains in force for both
arms. Complete layout always requires the baseline cache to remain absent and
the candidate cache bytes and mtime to remain unchanged.
Registered model verification remains mandatory in every receipt mode. Timing is
reported, not assumed to improve: paired candidate/baseline warm ratios are
descriptive and do not qualify a promotion or competitive result.
Raw logs and a command journal survive failure. receipt.json is written only
after every check succeeds; failure.json never qualifies a measurement result.

Harness-only tests: python3 scripts/test_benchmark_fast_answer_cache.py
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time


ARMS = ("baseline", "candidate")
MODEL_ID = "potion-multilingual-128m"
MODEL_DIRECTORY = "potion-multilingual-128M"
MODEL_REPOSITORY = "minishlab/potion-multilingual-128M"
MODEL_REVISION = "a28f4eebecd4dc585034f605e52d414878a0417c"
MODEL_ARTIFACTS = {
    "tokenizer.json": {
        "size": 18616131,
        "sha256": "19f1909063da3cfe3bd83a782381f040dccea475f4816de11116444a73e1b6a1",
    },
    "model.safetensors": {
        "size": 512361560,
        "sha256": "14b5eb39cb4ce5666da8ad1f3dc6be4346e9b2d601c073302fa0a31bf7943397",
    },
}
QUERY = "staggered reconnects after brief outages"
SCENARIOS = ("semantic-hits", "filtered-empty")
LAYOUTS = ("legacy", "complete")
EMPTY_FILTER = "ext:fsfs-cache-no-match"
DOCUMENTS = {
    "retry.md": "Recover transient network failures with exponential backoff, bounded retries, and random jitter.",
    "gardening.md": "Tomato seedlings need warm soil, steady sunlight, compost, and careful watering.",
    "accounting.md": "Reconcile invoices against purchase orders before closing the monthly ledger.",
    "astronomy.md": "A telescope gathers faint starlight so astronomers can study distant galaxies.",
    "baking.md": "Bread develops flavor through a slow fermentation before baking in a hot oven.",
    "music.md": "A string quartet balances melody, harmony, rhythm, and dynamics across four players.",
    "fitness.md": "Progressive resistance training strengthens muscles when recovery and nutrition are adequate.",
    "legal.md": "A written contract records the parties, obligations, remedies, and governing law.",
    "database.md": "A database transaction groups related updates into one atomic durable operation.",
    "typography.md": "Readable typography uses deliberate spacing, line length, contrast, and type hierarchy.",
}


class BenchmarkError(RuntimeError):
    """A failed precondition or measured correctness check."""


def require(condition, message):
    # These gates must still run with PYTHONOPTIMIZE or python -O.
    if not condition:
        raise BenchmarkError(message)


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("baseline", "baseline-source", "candidate", "candidate-source",
                 "models", "build-metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=240)
    parser.add_argument("--receipt-mode", choices=("valid", "absent", "malformed"),
                        default="valid")
    parser.add_argument("--scenario", choices=SCENARIOS, default="semantic-hits")
    parser.add_argument("--layout", choices=LAYOUTS, default="legacy")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    return args


def validate_output_location(args):
    require(not args.output.exists(), "output directory already exists")
    for root in (args.baseline_source, args.candidate_source, args.models):
        require(not args.output.is_relative_to(root),
                f"output must be outside input source/model roots: {root}")


def command_feature_profile(command):
    require(Path(command[0]).name == "cargo", "build command must directly invoke cargo")
    arguments = command[1:]
    if arguments and arguments[0].startswith("+"):
        arguments = arguments[1:]
    require(arguments and arguments[0] == "build", "metadata must record cargo build")
    features = set()
    profile = None
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        require(not (argument.startswith(("-q", "-v", "-r")) and len(argument) > 2),
                "use separate short Cargo options to record an unambiguous feature/profile")
        if argument in ("--release", "-r"):
            require(profile is None, "build command selects more than one profile")
            profile = "release"
        elif argument == "--profile" or argument.startswith("--profile="):
            if argument == "--profile":
                index += 1
                require(index < len(arguments), "--profile needs a value")
                value = arguments[index]
            else:
                value = argument.partition("=")[2]
            require(profile is None and value and not value.startswith("-"),
                    "build command has invalid or conflicting profile flags")
            profile = value
        elif argument in ("--features", "-F") or argument.startswith(("--features=", "-F")):
            if argument in ("--features", "-F"):
                index += 1
                require(index < len(arguments), "--features needs a value")
                value = arguments[index]
            elif argument.startswith("--features="):
                value = argument.partition("=")[2]
            else:
                value = argument[2:]
            require(value and not value.startswith("-"), "build command has empty feature flags")
            features.update(part for part in re.split(r"[,\s]+", value) if part)
        index += 1
    return {
        "cargo_profile": profile or "dev",
        "default_features": "--no-default-features" not in arguments,
        "all_features": "--all-features" in arguments,
        "features": sorted(features),
    }


def load_build_metadata(path):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(metadata, dict) and type(metadata.get("schema_version")) is int
            and metadata["schema_version"] == 1, "build metadata must use schema_version 1")
    command = metadata.get("command")
    require(isinstance(command, list) and command
            and all(isinstance(part, str) and part for part in command),
            "build metadata needs the actual common command as an argv array")
    selection = command_feature_profile(command)
    recorded_selection = metadata.get("feature_profile")
    require(isinstance(recorded_selection, dict)
            and type(recorded_selection.get("default_features")) is bool
            and type(recorded_selection.get("all_features")) is bool
            and recorded_selection == selection,
            "build metadata feature_profile differs from the actual command flags")
    environment = metadata.get("environment")
    require(isinstance(environment, dict)
            and all(isinstance(key, str) and isinstance(value, str)
                    for key, value in environment.items()),
            "build metadata needs the actual common environment map")
    rustc = metadata.get("rustc_vv")
    require(isinstance(rustc, str) and rustc.startswith("rustc ")
            and all(any(line.startswith(field + ": ") for line in rustc.splitlines())
                    for field in ("binary", "commit-hash", "commit-date", "host", "release",
                                  "LLVM version")),
            "build metadata needs complete actual rustc -Vv output")
    require(isinstance(metadata.get("arms"), dict) and set(metadata["arms"]) == set(ARMS),
            "build metadata must describe both baseline and candidate")
    for arm in ARMS:
        entry = metadata["arms"][arm]
        require(isinstance(entry, dict) and type(entry.get("build_exit_code")) is int
                and entry["build_exit_code"] == 0,
                f"{arm}: metadata must record a successful build")
        for key, length in (("source_head", 40), ("tracked_diff_sha256", 64),
                            ("binary_sha256", 64)):
            value = entry.get(key)
            require(isinstance(value, str) and len(value) == length
                    and all(char in "0123456789abcdef" for char in value),
                    f"{arm}: invalid or missing {key}")
    return metadata


def git_bytes(root, *arguments):
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *arguments],
        check=False, capture_output=True, timeout=30,
    )
    require(result.returncode == 0,
            f"cannot inspect source {root}: {result.stderr.decode(errors='replace')}")
    return result.stdout


def validate_source(root, recorded):
    actual_root = Path(git_bytes(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    require(actual_root == root, f"source must name the Git worktree root: {root}")
    head = git_bytes(root, "rev-parse", "HEAD").decode().strip()
    require(head == recorded["source_head"], f"source HEAD differs from successful build: {root}")
    diff = git_bytes(root, "diff", "HEAD", "--binary", "--")
    diff_sha256 = hashlib.sha256(diff).hexdigest()
    require(diff_sha256 == recorded["tracked_diff_sha256"],
            f"tracked source diff differs from successful build: {root}")
    status = git_bytes(root, "status", "--porcelain=v1", "--untracked-files=all")
    require(not diff and not status,
            f"source must be clean, including non-ignored untracked files: {root}")
    return {"root": str(root), "head": head, "tracked_diff_sha256": diff_sha256,
            "clean": True}


def validate_binary(path, expected_sha256):
    require(path.is_file() and os.access(path, os.X_OK), f"binary is not executable: {path}")
    require(sha256_file(path) == expected_sha256,
            f"binary differs from successful build metadata: {path}")


def verify_file(path, expected):
    require(path.is_file(), f"model artifact is missing: {path}")
    require(path.stat().st_size == expected["size"], f"model artifact size differs: {path}")
    require(sha256_file(path) == expected["sha256"], f"model artifact SHA-256 differs: {path}")


def private_inputs(args, metadata):
    binaries = {}
    binary_root = args.output / "binaries"
    binary_root.mkdir()
    for arm in ARMS:
        source = getattr(args, arm)
        destination = binary_root / f"{arm}-fsfs"
        shutil.copy2(source, destination)
        validate_binary(destination, metadata["arms"][arm]["binary_sha256"])
        binaries[arm] = destination
    install = args.output / "models" / MODEL_DIRECTORY
    install.mkdir(parents=True)
    for name, expected in MODEL_ARTIFACTS.items():
        source = args.models / MODEL_DIRECTORY / name
        verify_file(source, expected)
        destination = install / name
        shutil.copyfile(source, destination)
        verify_file(destination, expected)
    return binaries


def runtime_environment(output, layout="legacy"):
    require(layout in LAYOUTS, "unknown index layout")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("FSFS_", "FRANKENSEARCH_"))}
    for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        environment.pop(name, None)
    for name, directory in (("HOME", "home"), ("XDG_CONFIG_HOME", "config"),
                            ("XDG_CACHE_HOME", "cache"), ("XDG_DATA_HOME", "data")):
        path = output / directory
        path.mkdir()
        environment[name] = str(path)
    environment.update(
        FRANKENSEARCH_MODEL_DIR=str(output / "models"),
        FRANKENSEARCH_OFFLINE="1", FRANKENSEARCH_ALLOW_DOWNLOAD="0",
        FSFS_COMPLETE_GENERATIONS=str(int(layout == "complete")),
        RUST_LOG="info", FRANKENSEARCH_LOG="info",
        NO_COLOR="1", RAYON_NUM_THREADS="1",
    )
    return environment


class CommandRunner:
    def __init__(self, output, environment, timeout):
        self.output = output
        self.environment = environment
        self.timeout = timeout
        self.events = []

    def invoke(self, binary, label, command):
        started = time.perf_counter_ns()
        timed_out = False
        launch_error = None
        try:
            outcome = subprocess.run(
                [str(binary), *command], cwd=self.output, env=self.environment,
                capture_output=True, timeout=self.timeout,
            )
            stdout, stderr, exit_code = outcome.stdout, outcome.stderr, outcome.returncode
        except subprocess.TimeoutExpired as error:
            stdout, stderr, exit_code = error.stdout or b"", error.stderr or b"", None
            timed_out = True
        except OSError as error:
            stdout, stderr, exit_code = b"", str(error).encode(), None
            launch_error = str(error)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        (self.output / f"{label}.stdout.log").write_bytes(stdout)
        (self.output / f"{label}.stderr.log").write_bytes(stderr)
        event = {
            "label": label, "binary": str(binary), "command": command,
            "evidence_class": "diagnostic-maintenance",
            "exit_code": exit_code, "timed_out": timed_out, "launch_error": launch_error,
            "elapsed_ms": elapsed_ms,
            "model_loads": stderr.count(b"Model2Vec model loaded"),
            "receipt_acceptances": stderr.count(b"model verification receipt accepted"),
        }
        self.events.append(event)
        with (self.output / "commands.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(event, sort_keys=True) + "\n")
        print(json.dumps(event, sort_keys=True), flush=True)
        require(not timed_out, f"{label}: timed out; partial output retained")
        require(launch_error is None, f"{label}: could not execute binary; diagnostic retained")
        require(exit_code == 0, f"{label}: exit {exit_code}; output retained")
        try:
            response = json.loads(stdout)
        except (ValueError, UnicodeError) as error:
            raise BenchmarkError(f"{label}: stdout is not valid JSON") from error
        require(isinstance(response, dict) and response.get("ok") is True,
                f"{label}: response does not report success")
        return {**event, "response": response}


def prepare_receipt_mode(output, mode):
    require(mode in ("valid", "absent", "malformed"), "unknown receipt mode")
    path = output / "models" / MODEL_DIRECTORY / ".verified"
    require(path.is_file() and path.stat().st_size > 0,
            "baseline CLI verification did not create a private model receipt")
    verified_sha256 = sha256_file(path)
    preserved = path
    if mode != "valid":
        preserved = output / "verified-model-receipt.json"
        require(not preserved.exists(), "verification receipt archive already exists")
        path.rename(preserved)
        if mode == "malformed":
            path.write_bytes(b"{malformed-verification-receipt\n")
    return {
        "mode": mode, "preserved_path": str(preserved),
        "verified_sha256": verified_sha256,
        "expected_sha256": sha256_file(path) if path.exists() else None,
        "expected_mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
    }


def validate_receipt_state(output, state):
    preserved = Path(state["preserved_path"])
    require(preserved.is_file() and sha256_file(preserved) == state["verified_sha256"],
            "the original verified model receipt changed")
    path = output / "models" / MODEL_DIRECTORY / ".verified"
    if state["mode"] == "absent":
        require(not path.exists() and not path.is_symlink(),
                "indexing/search unexpectedly minted a model receipt")
    else:
        require(path.is_file() and sha256_file(path) == state["expected_sha256"]
                and path.stat().st_mtime_ns == state["expected_mtime_ns"],
                "indexing/search changed or repaired the model receipt")


def setup_fixture(runner, binaries, output, receipt_mode, layout="legacy"):
    require(layout in LAYOUTS, "unknown index layout")
    corpus = output / "corpus"
    corpus.mkdir()
    for name, contents in DOCUMENTS.items():
        (corpus / name).write_text(contents, encoding="utf-8")
    config = output / "fsfs.toml"
    config.write_text(
        f'[indexing]\nfast_model = "{MODEL_ID}"\n[search]\nfast_only = true\n',
        encoding="utf-8",
    )
    receipt_path = output / "models" / MODEL_DIRECTORY / ".verified"
    require(not receipt_path.exists(), "private model install unexpectedly has a receipt")
    verified = runner.invoke(
        binaries["baseline"], "setup-verify-model",
        ["download-models", MODEL_ID, "--offline", "--verify", "--format", "json"],
    )
    data = verified["response"].get("data", {})
    require(isinstance(data, dict) and data.get("operation") == "verify",
            "baseline CLI did not report a model verification operation")
    entries = data.get("models")
    require(isinstance(entries, list) and len(entries) == 1 and isinstance(entries[0], dict)
            and entries[0].get("id") == MODEL_ID and entries[0].get("state") == "verified"
            and entries[0].get("verified") is True,
            "baseline CLI did not verify the private registered Potion installation")
    receipt_state = prepare_receipt_mode(output, receipt_mode)
    template = output / "template-index"
    runner.invoke(
        binaries["baseline"], "setup-index",
        ["index", str(corpus), "--config", str(config), "--index-dir", str(template),
         "--fast-only", "--format", "json"],
    )
    require(template.is_dir(), "baseline indexing did not create the template index")
    if layout == "complete":
        complete_generation_snapshot(template)
    else:
        require(not (template / "FSFS-CURRENT").exists(),
                "legacy indexing unexpectedly selected a complete generation")
    validate_receipt_state(output, receipt_state)
    return template, config, receipt_state


def complete_generation_snapshot(index):
    """Bind measurement to the selected bundle without trusting cache contents."""
    pointer = index / "FSFS-CURRENT"
    require(pointer.is_file() and not pointer.is_symlink() and pointer.stat().st_size <= 256,
            f"complete layout requires a bounded regular FSFS-CURRENT: {index}")
    pointer_bytes = pointer.read_bytes()
    try:
        fields = pointer_bytes.decode("utf-8").split("\n")
    except UnicodeError as error:
        raise BenchmarkError(f"invalid complete-generation pointer: {index}") from error
    framing_valid = (
        len(fields) == 4 and fields[0] == "FSFS-COMPLETE-GENERATION-v1"
        or len(fields) == 5 and fields[0] == "FSFS-COMPLETE-GENERATION-v2"
        and re.fullmatch(r"[0-9a-f]{64}", fields[3]) is not None
    )
    require(framing_valid and fields[-1] == ""
            and re.fullmatch(r"g-[0-9a-f]{32}-[0-9a-f]{8}-[0-9a-f]{16}", fields[1])
            and re.fullmatch(r"[0-9a-f]{64}", fields[2]),
            f"invalid complete-generation pointer: {index}")
    generation = {"id": fields[1], "manifest_sha256": fields[2]}
    generations = index / "generations"
    bundle = generations / generation["id"]
    require(generations.is_dir() and not generations.is_symlink()
            and bundle.is_dir() and not bundle.is_symlink(),
            f"selected complete-generation bundle is missing or indirect: {index}")
    require(not (bundle / "query_cache").exists() and not (bundle / "query_cache").is_symlink(),
            "answer caches must remain outside the sealed complete-generation bundle")
    inventory = {}
    for path in sorted(bundle.rglob("*")):
        require(not path.is_symlink(), f"sealed bundle contains a symlink: {path}")
        relative = path.relative_to(bundle).as_posix()
        if path.is_dir():
            inventory[relative] = {"kind": "directory"}
        else:
            require(path.is_file(), f"sealed bundle contains a non-regular artifact: {path}")
            stamp = path.stat()
            inventory[relative] = {
                "kind": "file", "sha256": sha256_file(path), "size": stamp.st_size,
                "mtime_ns": stamp.st_mtime_ns,
            }
    manifest = inventory.get("FSFS-BUNDLE.json", {})
    require(manifest.get("sha256") == generation["manifest_sha256"],
            f"selected complete-generation manifest digest differs: {index}")
    return {"generation": generation, "inventory": inventory,
            "pointer_sha256": hashlib.sha256(pointer_bytes).hexdigest(),
            "pointer_mtime_ns": pointer.stat().st_mtime_ns}


def validate_unchanged_generation(index, before):
    require(complete_generation_snapshot(index) == before,
            f"selected complete-generation pointer or sealed bundle changed during search: {index}")


def cache_snapshot(index, expected_records=1, complete_generation=None):
    require(expected_records in (0, 1), "unsupported expected cache record count")
    records = sorted((index / "query_cache").glob("*.json"))
    require(len(records) == expected_records,
            f"expected {expected_records} persisted answer(s), found {len(records)}: {index}")
    if not records:
        return {"names": [], "bytes": None, "mtime_ns": None}
    path = records[0]
    contents = path.read_bytes()
    if complete_generation is not None:
        try:
            record = json.loads(contents)
        except (ValueError, UnicodeError) as error:
            raise BenchmarkError(f"invalid complete-generation cache record: {path}") from error
        require(isinstance(record, dict)
                and record.get("complete_generation") == complete_generation,
                f"cached answer does not name the exact selected complete generation: {path}")
    return {"names": [path.name], "bytes": contents, "mtime_ns": path.stat().st_mtime_ns}


def semantic_hits(data, label):
    require(isinstance(data, dict), f"{label}: semantic response has no payload")
    hits = data.get("hits")
    require(isinstance(hits, list) and hits, f"{label}: semantic search returned no hits")
    require(any(isinstance(hit, dict) and hit.get("semantic_rank") is not None for hit in hits)
            and data.get("vector_generation_is_hash") is not True,
            f"{label}: result did not exercise the semantic lane")
    return hits


def search_command(config, index, scenario):
    require(scenario in SCENARIOS, "unknown benchmark scenario")
    command = [
        "search", QUERY, "--config", str(config), "--index-dir", str(index),
        "--no-daemon", "--fast-only", "--limit", "3", "--format", "json",
    ]
    if scenario == "filtered-empty":
        command.extend(["--filter", EMPTY_FILTER])
    return command


def run_semantic_controls(runner, binaries, output, template, config, receipt_state,
                          layout="legacy"):
    require(layout in LAYOUTS, "unknown index layout")
    controls = {}
    reference_hits = None
    reference_generation = None
    for arm in ARMS:
        index = output / f"{arm}-semantic-control-index"
        shutil.copytree(template, index)
        require(not list((index / "query_cache").glob("*.json")),
                "semantic control must start without a persisted answer")
        sealed = complete_generation_snapshot(index) if layout == "complete" else None
        validate_receipt_state(output, receipt_state)
        result = runner.invoke(
            binaries[arm], f"{arm}-semantic-control",
            search_command(config, index, "semantic-hits"),
        )
        validate_receipt_state(output, receipt_state)
        if sealed is not None:
            validate_unchanged_generation(index, sealed)
            cache_snapshot(index, int(arm == "candidate"), sealed["generation"])
        require(result["model_loads"] == 1,
                f"{arm}: semantic control must load Model2Vec exactly once")
        data = result["response"].get("data")
        hits = semantic_hits(data, f"{arm} control")
        require(data.get("query") == QUERY and data.get("phase") == "initial"
                and type(data.get("returned_hits")) is int and data["returned_hits"] == len(hits),
                f"{arm}: semantic control did not answer the requested query")
        generation = data.get("vector_generation_id")
        require(isinstance(generation, str) and generation,
                f"{arm}: semantic control has no producer identity")
        require(reference_hits is None or hits == reference_hits,
                "semantic control ordered hits/scores differ across arms")
        require(reference_generation is None or generation == reference_generation,
                "semantic control producer identities differ across arms")
        reference_hits, reference_generation = hits, generation
        controls[arm] = {**result, "vector_generation_id": generation,
                        "layout": layout,
                        "complete_generation": sealed["generation"] if sealed else None}
    return controls


def validate_pair(arm, cold, warm, before, after, reference_hits, receipt_mode,
                  scenario="semantic-hits", expected_generation_id=None, layout="legacy"):
    require(scenario in SCENARIOS, "unknown benchmark scenario")
    require(layout in LAYOUTS, "unknown index layout")
    cold_data, warm_data = cold["response"].get("data"), warm["response"].get("data")
    require(isinstance(cold_data, dict) and cold_data == warm_data,
            f"{arm}: warm cached payload differs from cold retrieval")
    require(before["names"] == after["names"], f"{arm}: warm cache record count/name changed")
    require(before["bytes"] == after["bytes"], f"{arm}: warm cache bytes changed")
    if layout == "complete":
        expected = int(arm == "candidate")
        require(len(before["names"]) == expected
                and (before["bytes"] is not None) == bool(expected)
                and (before["mtime_ns"] is not None) == bool(expected),
                f"{arm}: complete layout requires {expected} persisted answer(s)")
    if layout == "legacy" and scenario == "filtered-empty" and arm == "baseline":
        require(before["mtime_ns"] != after["mtime_ns"],
                "baseline: filtered-empty regression must recompute and rewrite the warm answer")
    else:
        require(before["mtime_ns"] == after["mtime_ns"], f"{arm}: warm cache was rewritten")
    require(cold["model_loads"] == 1, f"{arm}: cold search must load Model2Vec exactly once")
    require(warm["model_loads"] == (1 if arm == "baseline" else 0),
            f"{arm}: unexpected warm Model2Vec load count")
    if receipt_mode == "valid":
        require(warm["receipt_acceptances"] > 0,
                f"{arm}: warm search did not accept the model verification receipt")
    else:
        require(warm["receipt_acceptances"] == 0,
                f"{arm}: warm search accepted an absent or malformed receipt")
    if scenario == "filtered-empty":
        require(isinstance(expected_generation_id, str) and expected_generation_id,
                "filtered-empty measurement requires an unfiltered semantic control")
        hits = cold_data.get("hits")
        require(hits == [] and type(cold_data.get("returned_hits")) is int
                and cold_data["returned_hits"] == 0 and cold_data.get("phase") == "initial"
                and cold_data.get("query") == QUERY,
                f"{arm}: filtered-empty request did not return the expected empty Initial")
        require(cold_data.get("vector_generation_id") == expected_generation_id
                and cold_data.get("vector_generation_is_hash") is not True,
                f"{arm}: filtered-empty producer differs from its semantic control")
    else:
        hits = semantic_hits(cold_data, arm)
    require(reference_hits is None or hits == reference_hits,
            f"{arm}: ordered hits/scores differ across arms or rounds")
    return hits


def run_rounds(runner, binaries, output, template, config, rounds, receipt_state,
               scenario="semantic-hits", semantic_controls=None, layout="legacy"):
    require(scenario in SCENARIOS, "unknown benchmark scenario")
    require(layout in LAYOUTS, "unknown index layout")
    if scenario == "filtered-empty":
        require(isinstance(semantic_controls, dict) and set(semantic_controls) == set(ARMS),
                "filtered-empty measurement requires both semantic controls")
    rows = []
    reference_hits = None
    reference_payload = None
    for ordinal in range(rounds):
        for arm in ARMS if ordinal % 2 == 0 else reversed(ARMS):
            index = output / f"{arm}-{ordinal}-index"
            shutil.copytree(template, index)
            require(not list((index / "query_cache").glob("*.json")),
                    f"{arm}: cold request already has a persisted answer")
            sealed = complete_generation_snapshot(index) if layout == "complete" else None
            expected_records = int(layout == "legacy" or arm == "candidate")
            generation = sealed["generation"] if sealed else None
            command = search_command(config, index, scenario)
            validate_receipt_state(output, receipt_state)
            cold = runner.invoke(binaries[arm], f"{arm}-{ordinal}-cold", command)
            validate_receipt_state(output, receipt_state)
            if sealed is not None:
                validate_unchanged_generation(index, sealed)
            before = cache_snapshot(index, expected_records, generation)
            warm = runner.invoke(binaries[arm], f"{arm}-{ordinal}-warm", command)
            validate_receipt_state(output, receipt_state)
            if sealed is not None:
                validate_unchanged_generation(index, sealed)
            after = cache_snapshot(index, expected_records, generation)
            reference_hits = validate_pair(
                arm, cold, warm, before, after, reference_hits, receipt_state["mode"], scenario,
                semantic_controls[arm]["vector_generation_id"] if semantic_controls else None,
                layout)
            if scenario == "filtered-empty":
                payload = cold["response"]["data"]
                require(reference_payload is None or payload == reference_payload,
                        "filtered-empty payload differs across arms or rounds")
                reference_payload = payload
            for phase, result in (("cold", cold), ("warm", warm)):
                rows.append({
                    **{key: value for key, value in result.items() if key != "response"},
                    "arm": arm, "round": ordinal, "phase": phase,
                    "receipt_mode": receipt_state["mode"], "scenario": scenario, "layout": layout,
                    "cache_sha256": (hashlib.sha256(before["bytes"]).hexdigest()
                                     if before["bytes"] is not None else None),
                    "warm_payload_identical": True, "cache_bytes_unchanged": True,
                    "cache_mtime_unchanged": before["mtime_ns"] == after["mtime_ns"],
                    "cache_record_count": len(before["names"]),
                    "complete_generation": generation,
                    "sealed_bundle_unchanged": True if sealed else None,
                    "sealed_bundle_inventory": sealed["inventory"] if sealed else None,
                    "selection_pointer_sha256": sealed["pointer_sha256"] if sealed else None,
                })
    return rows


def summarize(rows, rounds):
    expected = {(arm, ordinal, phase) for arm in ARMS for ordinal in range(rounds)
                for phase in ("cold", "warm")}
    require(len(rows) == len(expected)
            and {(row["arm"], row["round"], row["phase"]) for row in rows} == expected,
            "measurement rows are incomplete or duplicated")
    require(all(math.isfinite(row["elapsed_ms"]) and row["elapsed_ms"] > 0 for row in rows),
            "measurement elapsed times must be finite and positive")
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for phase in ("cold", "warm"):
            values = [row["elapsed_ms"] for row in rows
                      if row["arm"] == arm and row["phase"] == phase]
            summary[arm][phase] = {"samples_ms": values, "median_ms": statistics.median(values)}
    ratios = []
    for ordinal in range(rounds):
        paired = {row["arm"]: row["elapsed_ms"] for row in rows
                  if row["round"] == ordinal and row["phase"] == "warm"}
        ratios.append(paired["candidate"] / paired["baseline"])
    summary["paired_warm_candidate_over_baseline"] = {
        "samples": ratios, "median": statistics.median(ratios),
        "minimum": min(ratios), "maximum": max(ratios),
    }
    return summary


def run_benchmark(args):
    validate_output_location(args)
    args.output.mkdir(parents=True, exist_ok=False)
    runner = None
    metadata = None
    sources = {}
    try:
        metadata = load_build_metadata(args.build_metadata)
        for arm in ARMS:
            sources[arm] = validate_source(getattr(args, arm + "_source"), metadata["arms"][arm])
            validate_binary(getattr(args, arm), metadata["arms"][arm]["binary_sha256"])
        for name, expected in MODEL_ARTIFACTS.items():
            verify_file(args.models / MODEL_DIRECTORY / name, expected)
        binaries = private_inputs(args, metadata)
        environment = runtime_environment(args.output, args.layout)
        runner = CommandRunner(args.output, environment, args.timeout_seconds)
        template, config, receipt_state = setup_fixture(
            runner, binaries, args.output, args.receipt_mode, args.layout)
        semantic_controls = run_semantic_controls(
            runner, binaries, args.output, template, config, receipt_state, args.layout
        ) if args.scenario == "filtered-empty" else None
        rows = run_rounds(
            runner, binaries, args.output, template, config, args.rounds, receipt_state,
            args.scenario, semantic_controls, args.layout)
        summary = summarize(rows, args.rounds)
        for arm in ARMS:
            require(validate_source(getattr(args, arm + "_source"), metadata["arms"][arm])
                    == sources[arm], f"{arm}: source provenance changed during measurement")
            for binary in (getattr(args, arm), binaries[arm]):
                validate_binary(binary, metadata["arms"][arm]["binary_sha256"])
        install = args.output / "models" / MODEL_DIRECTORY
        for name, expected in MODEL_ARTIFACTS.items():
            verify_file(install / name, expected)
        validate_receipt_state(args.output, receipt_state)
        receipt = {
            "schema_version": 1, "status": "completed",
            "evidence_class": "diagnostic-maintenance",
            "qualifies_promotion": False, "qualifies_competitive_win": False,
            "feature_profile": metadata["feature_profile"],
            "receipt_mode": args.receipt_mode, "scenario": args.scenario, "layout": args.layout,
            "filter": EMPTY_FILTER if args.scenario == "filtered-empty" else None,
            "semantic_controls": semantic_controls,
            "cross_arm_equivalence": ("complete-search-payloads"
                                      if args.scenario == "filtered-empty"
                                      else "ordered-hits-and-scores"),
            "cache_scope": (
                "persisted legacy-layout answer cache (FSFS_COMPLETE_GENERATIONS=0); "
                "fresh --no-daemon processes; no OS page cache eviction"
                if args.layout == "legacy" else
                "store-owned complete-generation answer cache (FSFS_COMPLETE_GENERATIONS=1); "
                "baseline has no persisted answers; candidate bound to the exact sealed bundle; "
                "fresh --no-daemon processes; no OS page cache eviction"
            ),
            "build_metadata": metadata, "source_provenance": sources,
            "model": {"id": MODEL_ID, "repository": MODEL_REPOSITORY,
                      "revision": MODEL_REVISION, "artifacts": MODEL_ARTIFACTS,
                      "receipt_state": receipt_state},
            "runtime_environment": {key: environment[key] for key in (
                "HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
                "FRANKENSEARCH_MODEL_DIR", "FRANKENSEARCH_OFFLINE",
                "FRANKENSEARCH_ALLOW_DOWNLOAD", "FSFS_COMPLETE_GENERATIONS",
                "RAYON_NUM_THREADS", "RUST_LOG", "FRANKENSEARCH_LOG", "NO_COLOR",
            )},
            "runtime_linkage_environment": {key: environment.get(key) for key in
                                            ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
                                             "LD_PRELOAD", "ORT_DYLIB_PATH")},
            "query": QUERY, "corpus_documents": DOCUMENTS, "rounds": args.rounds,
            "rows": rows, "summary": summary, "identical_hits_across_arms": True,
        }
        write_json(args.output / "receipt.json", receipt)
        print(json.dumps({
            "evidence_class": "diagnostic-maintenance",
            "qualifies_promotion": False, "qualifies_competitive_win": False,
            "feature_profile": metadata["feature_profile"], "receipt_mode": args.receipt_mode,
            "scenario": args.scenario, "layout": args.layout,
            "source_heads": {arm: sources[arm]["head"] for arm in ARMS}, "summary": summary,
        }, indent=2, sort_keys=True), flush=True)
        return receipt
    except BaseException as error:
        try:
            write_json(args.output / "failure.json", {
                "status": "failed", "evidence_class": "diagnostic-maintenance",
                "qualifies_measurement_result": False, "receipt_mode": args.receipt_mode,
                "scenario": args.scenario, "layout": args.layout,
                "build_metadata": metadata, "source_provenance": sources,
                "error_type": type(error).__name__, "error": str(error),
                "completed_commands": runner.events if runner else [],
            })
        except OSError as recording_error:
            print(f"Could not save failure metadata: {recording_error}", file=sys.stderr)
        raise


def main(argv=None):
    args = parse_args(argv)
    try:
        run_benchmark(args)
    except (BenchmarkError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Diagnostic benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
