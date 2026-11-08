#!/usr/bin/env python3
"""Harness-only regressions; these do not measure or qualify fsfs.

The fake protocol responses below exercise rejection and orchestration only.
They never enter the successful run_benchmark path or produce a result receipt.
No model is loaded and no fsfs executable is invoked. Temporary fixtures are
intentionally retained, matching the repository's no-deletion rule.
"""

from contextlib import chdir, redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import benchmark_fast_answer_cache as bench


EMPTY_DIFF = hashlib.sha256(b"").hexdigest()
FIXTURE_RESPONSE = {
    "ok": True,
    "data": {
        "hits": [{"path": "fixture-only/retry.md", "semantic_rank": 0, "score": 0.5}],
        "vector_generation_is_hash": False,
    },
}


def metadata_fixture():
    return {
        "schema_version": 1,
        "command": ["cargo", "build", "--locked", "--release", "-p", "frankensearch-fsfs",
                    "--bin", "fsfs", "--no-default-features", "--features", "semantic-support"],
        "environment": {},
        "feature_profile": {
            "cargo_profile": "release", "default_features": False,
            "all_features": False, "features": ["semantic-support"],
        },
        "rustc_vv": (
            "rustc 0.0.0 (fixture-only)\nbinary: rustc\ncommit-hash: fixture-only\n"
            "commit-date: fixture-only\nhost: fixture-only\nrelease: 0.0.0\nLLVM version: 0.0\n"
        ),
        "arms": {
            arm: {"source_head": "a" * 40, "tracked_diff_sha256": EMPTY_DIFF,
                  "binary_sha256": "b" * 64, "build_exit_code": 0}
            for arm in bench.ARMS
        },
    }


class FixtureRunner:
    """In-memory protocol stand-in, intentionally incapable of writing a result receipt."""

    def __init__(self, output, mode):
        self.output = output
        self.mode = mode
        self.calls = []

    def invoke(self, binary, label, command):
        self.calls.append((binary, label, command))
        response = copy.deepcopy(FIXTURE_RESPONSE)
        if label == "setup-verify-model":
            (self.output / "models" / bench.MODEL_DIRECTORY / ".verified").write_bytes(
                b"fixture-only verification marker"
            )
            response = {"ok": True, "data": {
                "operation": "verify",
                "models": [{"id": bench.MODEL_ID, "state": "verified", "verified": True}],
            }}
        elif label == "setup-index":
            Path(command[command.index("--index-dir") + 1]).mkdir()
        elif label.endswith("-cold"):
            index = Path(command[command.index("--index-dir") + 1])
            (index / "query_cache").mkdir()
            (index / "query_cache" / "fixture-only.json").write_bytes(b'{"fixture_only": true}')
        return {
            "response": response, "label": label, "elapsed_ms": 1,
            "model_loads": int(label.endswith("-cold") or label.startswith("baseline-")),
            "receipt_acceptances": int(self.mode == "valid"),
        }


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="fsfs-answer-cache-harness-test-")).resolve()

    def tearDown(self):
        self.assertEqual(list(self.root.rglob("receipt.json")), [],
                         "harness tests must never emit simulated measurement receipts")

    def arguments(self, **changes):
        values = {
            "baseline": "before-fsfs", "baseline-source": "before-source",
            "candidate": "after-fsfs", "candidate-source": "after-source",
            "models": "input-models", "build-metadata": "builds.json", "output": "output",
        }
        values.update(changes)
        return [part for name, value in values.items() if value is not None
                for part in ("--" + name, str(value))]

    def test_both_arms_and_sources_are_required(self):
        for missing in ("baseline", "baseline-source", "candidate", "candidate-source"):
            with self.subTest(missing=missing), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.parse_args(self.arguments(**{missing: None}))

    def test_paths_resolve_before_child_working_directory_changes(self):
        with chdir(self.root):
            args = bench.parse_args(self.arguments())
        for name in ("baseline", "baseline_source", "candidate", "candidate_source",
                     "models", "build_metadata", "output"):
            self.assertTrue(getattr(args, name).is_absolute())
            self.assertEqual(getattr(args, name).parent, self.root)
        self.assertEqual(args.receipt_mode, "valid")
        for option in (["--rounds", "0"], ["--timeout-seconds", "nan"],
                       ["--receipt-mode", "unknown"]):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.parse_args(self.arguments() + option)

    def test_output_cannot_overlap_inputs_or_reuse_previous_evidence(self):
        with chdir(self.root):
            args = bench.parse_args(self.arguments())
        bench.validate_output_location(args)
        for input_name in ("baseline_source", "candidate_source", "models"):
            args.output = getattr(args, input_name) / "results"
            with self.subTest(input_name=input_name):
                with self.assertRaisesRegex(bench.BenchmarkError, "outside"):
                    bench.validate_output_location(args)
        args.output = self.root
        with self.assertRaisesRegex(bench.BenchmarkError, "already exists"):
            bench.validate_output_location(args)

    def test_metadata_requires_successful_bound_builds_and_matching_feature_profile(self):
        metadata_path = self.root / "builds.json"
        valid = metadata_fixture()
        bench.write_json(metadata_path, valid)
        self.assertEqual(bench.load_build_metadata(metadata_path), valid)
        mutations = [
            lambda item: item["arms"].pop("candidate"),
            lambda item: item["arms"]["baseline"].update(build_exit_code=101),
            lambda item: item["arms"]["candidate"].update(binary_sha256="unrecorded"),
            lambda item: item.update(environment={"RUSTFLAGS": None}),
            lambda item: item.update(rustc_vv="rustc 0.0.0"),
            lambda item: item["feature_profile"].update(default_features=True),
            lambda item: item["feature_profile"].update(default_features=0),
            lambda item: item["feature_profile"].update(features=[]),
            lambda item: item.update(command=["echo", "planned cargo build"]),
        ]
        for ordinal, mutate in enumerate(mutations):
            invalid = copy.deepcopy(valid)
            mutate(invalid)
            bench.write_json(metadata_path, invalid)
            with self.subTest(mutation=ordinal):
                with self.assertRaises(bench.BenchmarkError):
                    bench.load_build_metadata(metadata_path)
        selection = bench.command_feature_profile([
            "/toolchain/cargo", "+nightly", "build", "--profile=bench-profile",
            "--no-default-features", "--features=semantic-support,native",
            "-F", "native", "--locked",
        ])
        self.assertEqual(selection, {
            "cargo_profile": "bench-profile", "default_features": False,
            "all_features": False, "features": ["native", "semantic-support"],
        })
        with self.assertRaisesRegex(bench.BenchmarkError, "separate short"):
            bench.command_feature_profile(["cargo", "build", "-qr"])

    def git(self, root, *arguments):
        environment = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
        result = subprocess.run(
            ["git", "-c", "user.name=Harness Test", "-c", "user.email=harness@example.invalid",
             "-C", str(root), *arguments], env=environment, capture_output=True, check=True,
        )
        return result.stdout

    def source_fixture(self):
        root = self.root / "source"
        root.mkdir()
        self.git(root, "init", "--initial-branch=main")
        (root / "source.rs").write_text("// fixture-only source\n", encoding="utf-8")
        self.git(root, "add", "source.rs")
        self.git(root, "commit", "-m", "fixture-only source")
        record = {"source_head": self.git(root, "rev-parse", "HEAD").decode().strip(),
                  "tracked_diff_sha256": EMPTY_DIFF}
        return root, record

    def test_a_different_clean_head_cannot_reuse_an_empty_diff_proof(self):
        root, record = self.source_fixture()
        bench.validate_source(root, record)
        (root / "source.rs").write_text("// a different fixture-only revision\n", encoding="utf-8")
        self.git(root, "add", "source.rs")
        self.git(root, "commit", "-m", "different fixture-only source")
        with self.assertRaisesRegex(bench.BenchmarkError, "HEAD differs"):
            bench.validate_source(root, record)

    def test_changed_tracked_or_untracked_sources_are_rejected(self):
        root, record = self.source_fixture()
        (root / "untracked.rs").write_text("// fixture-only untracked source\n", encoding="utf-8")
        with self.assertRaisesRegex(bench.BenchmarkError, "including non-ignored untracked"):
            bench.validate_source(root, record)
        (root / "source.rs").write_text("// fixture-only dirty source\n", encoding="utf-8")
        with self.assertRaisesRegex(bench.BenchmarkError, "diff differs"):
            bench.validate_source(root, record)
        record["tracked_diff_sha256"] = hashlib.sha256(
            self.git(root, "diff", "HEAD", "--binary", "--")).hexdigest()
        with self.assertRaisesRegex(bench.BenchmarkError, "must be clean"):
            bench.validate_source(root, record)

    def test_binary_and_model_artifact_hashes_are_enforced(self):
        path = self.root / "fixture-file"
        path.write_bytes(b"fixture-only")
        digest = bench.sha256_file(path)
        with self.assertRaisesRegex(bench.BenchmarkError, "not executable"):
            bench.validate_binary(path, digest)
        path.chmod(0o700)
        bench.validate_binary(path, digest)
        with self.assertRaisesRegex(bench.BenchmarkError, "differs"):
            bench.validate_binary(path, "0" * 64)
        bench.verify_file(path, {"size": path.stat().st_size, "sha256": digest})
        with self.assertRaisesRegex(bench.BenchmarkError, "size differs"):
            bench.verify_file(path, {"size": 1, "sha256": digest})
        with self.assertRaisesRegex(bench.BenchmarkError, "SHA-256 differs"):
            bench.verify_file(path, {"size": path.stat().st_size, "sha256": "0" * 64})

    def test_private_copy_never_copies_or_modifies_the_input_receipt(self):
        with chdir(self.root):
            args = bench.parse_args(self.arguments())
        args.output.mkdir()
        model = args.models / bench.MODEL_DIRECTORY
        model.mkdir(parents=True)
        external_receipt = model / ".verified"
        external_receipt.write_bytes(b"fixture-only external receipt")
        receipt_before = (external_receipt.read_bytes(), external_receipt.stat().st_mtime_ns)
        artifacts = {}
        for name in bench.MODEL_ARTIFACTS:
            (model / name).write_bytes(("fixture-only " + name).encode())
            artifacts[name] = {"size": (model / name).stat().st_size,
                               "sha256": bench.sha256_file(model / name)}
        (model / "unregistered.json").write_bytes(b"fixture-only extra file")
        metadata = metadata_fixture()
        for arm in bench.ARMS:
            path = getattr(args, arm)
            path.write_bytes(("fixture-only executable " + arm).encode())
            path.chmod(0o700)
            metadata["arms"][arm]["binary_sha256"] = bench.sha256_file(path)
        with mock.patch.object(bench, "MODEL_ARTIFACTS", artifacts):
            copied = bench.private_inputs(args, metadata)
        self.assertEqual(
            sorted(path.name for path in (args.output / "models" / bench.MODEL_DIRECTORY).iterdir()),
            sorted(artifacts),
        )
        self.assertEqual((external_receipt.read_bytes(), external_receipt.stat().st_mtime_ns),
                         receipt_before)
        for arm in bench.ARMS:
            self.assertEqual(copied[arm].read_bytes(), getattr(args, arm).read_bytes())
            self.assertNotEqual(copied[arm].stat().st_ino, getattr(args, arm).stat().st_ino)

    def pair(self, arm="candidate", mode="valid"):
        cold = {"response": copy.deepcopy(FIXTURE_RESPONSE), "model_loads": 1,
                "receipt_acceptances": int(mode == "valid")}
        warm = {**copy.deepcopy(cold), "model_loads": int(arm == "baseline")}
        cache = {"names": ["fixture-only.json"], "bytes": b"fixture-only", "mtime_ns": 10}
        return cold, warm, cache, copy.deepcopy(cache)

    def test_replay_gates_reject_hidden_work_and_cache_damage(self):
        mutations = [
            (lambda cold, warm, before, after: warm["response"]["data"].update(extra=True),
             "payload differs"),
            (lambda cold, warm, before, after: after.update(bytes=b"different"), "bytes changed"),
            (lambda cold, warm, before, after: after.update(mtime_ns=11), "rewritten"),
            (lambda cold, warm, before, after: after.update(names=["other.json"]), "count/name"),
            (lambda cold, warm, before, after: cold.update(model_loads=0), "cold search"),
            (lambda cold, warm, before, after: warm.update(model_loads=1), "warm Model2Vec"),
            (lambda cold, warm, before, after: warm.update(receipt_acceptances=0), "receipt"),
        ]
        for mutate, message in mutations:
            cold, warm, before, after = self.pair()
            mutate(cold, warm, before, after)
            with self.subTest(message=message):
                with self.assertRaisesRegex(bench.BenchmarkError, message):
                    bench.validate_pair("candidate", cold, warm, before, after, None, "valid")
        cold, warm, before, after = self.pair("baseline")
        warm["model_loads"] = 0
        with self.assertRaisesRegex(bench.BenchmarkError, "warm Model2Vec"):
            bench.validate_pair("baseline", cold, warm, before, after, None, "valid")
        cold, warm, before, after = self.pair()
        with self.assertRaisesRegex(bench.BenchmarkError, "across arms"):
            bench.validate_pair("candidate", cold, warm, before, after, [], "valid")
        for mode in ("absent", "malformed"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(bench.BenchmarkError, "absent or malformed"):
                    bench.validate_pair("candidate", cold, warm, before, after, None, mode)

    def test_no_hits_or_hash_control_cannot_pass_as_semantic_search(self):
        for data in ({"hits": []}, {"hits": [{"semantic_rank": None}]},
                     {**FIXTURE_RESPONSE["data"], "vector_generation_is_hash": True}):
            cold, warm, before, after = self.pair()
            cold["response"]["data"] = warm["response"]["data"] = data
            with self.subTest(data=data):
                with self.assertRaises(bench.BenchmarkError):
                    bench.validate_pair("candidate", cold, warm, before, after, None, "valid")

    def test_cache_snapshot_requires_exactly_one_persisted_record(self):
        index = self.root / "index"
        (index / "query_cache").mkdir(parents=True)
        with self.assertRaisesRegex(bench.BenchmarkError, "found 0"):
            bench.cache_snapshot(index)
        (index / "query_cache" / "one.json").write_bytes(b"fixture-only")
        self.assertEqual(bench.cache_snapshot(index)["names"], ["one.json"])
        (index / "query_cache" / "two.json").write_bytes(b"fixture-only")
        with self.assertRaisesRegex(bench.BenchmarkError, "found 2"):
            bench.cache_snapshot(index)

    def test_verify_precedes_index_and_paired_order_alternates_in_every_receipt_mode(self):
        for mode in ("valid", "absent", "malformed"):
            with self.subTest(mode=mode):
                output = self.root / mode
                (output / "models" / bench.MODEL_DIRECTORY).mkdir(parents=True)
                runner = FixtureRunner(output, mode)
                binaries = {arm: self.root / (arm + "-fixture-only") for arm in bench.ARMS}
                template, config, state = bench.setup_fixture(runner, binaries, output, mode)
                rows = bench.run_rounds(runner, binaries, output, template, config, 2, state)
                self.assertEqual([label for _, label, _ in runner.calls], [
                    "setup-verify-model", "setup-index",
                    "baseline-0-cold", "baseline-0-warm", "candidate-0-cold", "candidate-0-warm",
                    "candidate-1-cold", "candidate-1-warm", "baseline-1-cold", "baseline-1-warm",
                ])
                self.assertEqual(runner.calls[0], (
                    binaries["baseline"], "setup-verify-model",
                    ["download-models", bench.MODEL_ID, "--offline", "--verify", "--format", "json"],
                ))
                self.assertEqual(len(rows), 8)
                self.assertTrue(all(row["receipt_mode"] == mode for row in rows))
                with self.assertRaisesRegex(bench.BenchmarkError, "incomplete or duplicated"):
                    bench.summarize(rows[:-1], 2)
                invalid_rows = copy.deepcopy(rows)
                invalid_rows[-1] = copy.deepcopy(invalid_rows[0])
                with self.assertRaisesRegex(bench.BenchmarkError, "incomplete or duplicated"):
                    bench.summarize(invalid_rows, 2)
                # A minted/repaired or rewritten receipt must stop the next phase.
                path = output / "models" / bench.MODEL_DIRECTORY / ".verified"
                if mode == "valid":
                    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
                else:
                    path.write_bytes(b"unexpected replacement")
                with self.assertRaisesRegex(bench.BenchmarkError, "receipt"):
                    bench.validate_receipt_state(output, state)

    def test_verification_success_without_a_receipt_cannot_start_indexing(self):
        (self.root / "models" / bench.MODEL_DIRECTORY).mkdir(parents=True)
        runner = mock.Mock()
        runner.invoke.return_value = {"response": {"ok": True, "data": {
            "operation": "verify",
            "models": [{"id": bench.MODEL_ID, "state": "verified", "verified": True}],
        }}}
        with self.assertRaisesRegex(bench.BenchmarkError, "did not create"):
            bench.setup_fixture(runner, {"baseline": self.root / "fixture-only"}, self.root, "valid")
        self.assertEqual(runner.invoke.call_count, 1)

    def test_malformed_verification_protocol_cannot_start_indexing(self):
        cases = (None, [], {"operation": "verify", "models": "wrong type"},
                 {"operation": "verify", "models": [None]},
                 {"operation": "verify", "models": [
                     {"id": bench.MODEL_ID, "state": "verified", "verified": False},
                 ]})
        for ordinal, data in enumerate(cases):
            output = self.root / f"invalid-verify-{ordinal}"
            (output / "models" / bench.MODEL_DIRECTORY).mkdir(parents=True)
            runner = mock.Mock()
            runner.invoke.return_value = {"response": {"ok": True, "data": data}}
            with self.subTest(data=data):
                with self.assertRaises(bench.BenchmarkError):
                    bench.setup_fixture(runner, {"baseline": output / "fixture-only"},
                                        output, "valid")
            self.assertEqual(runner.invoke.call_count, 1)

    def test_timeout_preserves_partial_logs_and_failure_journal(self):
        runner = bench.CommandRunner(self.root, {}, 1)
        error = subprocess.TimeoutExpired("fixture-only", 1, output=b"partial output",
                                          stderr=b"partial error")
        with mock.patch.object(bench.subprocess, "run", side_effect=error):
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(bench.BenchmarkError, "timed out"):
                    runner.invoke(self.root / "fixture-only", "fixture-timeout", [])
        self.assertEqual((self.root / "fixture-timeout.stdout.log").read_bytes(), b"partial output")
        self.assertEqual((self.root / "fixture-timeout.stderr.log").read_bytes(), b"partial error")
        self.assertTrue(json.loads((self.root / "commands.jsonl").read_text())["timed_out"])

    def test_nonzero_invalid_json_and_unsuccessful_envelopes_are_rejected(self):
        cases = ((2, b'{"ok": true}'), (0, b"not-json"), (0, b'{"ok": false}'))
        for ordinal, (code, stdout) in enumerate(cases):
            runner = bench.CommandRunner(self.root, {}, 1)
            result = subprocess.CompletedProcess([], code, stdout=stdout, stderr=b"fixture-only")
            with mock.patch.object(bench.subprocess, "run", return_value=result):
                with redirect_stdout(io.StringIO()), self.subTest(case=ordinal):
                    with self.assertRaises(bench.BenchmarkError):
                        runner.invoke(self.root / "fixture-only", f"rejected-{ordinal}", [])

    def test_input_failure_preserves_a_failure_record_without_invoking_fsfs(self):
        with chdir(self.root):
            args = bench.parse_args(self.arguments())
        bench.write_json(args.build_metadata, {})
        with mock.patch.object(bench, "CommandRunner") as runner:
            with self.assertRaises(bench.BenchmarkError):
                bench.run_benchmark(args)
            runner.assert_not_called()
        failure = json.loads((args.output / "failure.json").read_text())
        self.assertEqual(failure["status"], "failed")
        self.assertFalse(failure["qualifies_measurement_result"])
        self.assertEqual(failure["completed_commands"], [])


if __name__ == "__main__":
    unittest.main()
