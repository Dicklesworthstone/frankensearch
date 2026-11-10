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
        "query": bench.QUERY, "phase": "initial", "returned_hits": 1,
        "hits": [{"path": "fixture-only/retry.md", "semantic_rank": 0, "score": 0.5}],
        "vector_generation_id": bench.MODEL_ID,
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


def complete_generation_fixture(index, version=2):
    """Only pointer framing/tree inventory; not a valid fsfs generation or evidence."""
    generation_id = "g-" + "a" * 32 + "-" + "b" * 8 + "-" + "c" * 16
    bundle = index / "generations" / generation_id
    bundle.mkdir(parents=True)
    manifest = bundle / "FSFS-BUNDLE.json"
    manifest.write_bytes(b'{"fixture_only": true}\n')
    (bundle / "vectors.fixture").write_bytes(b"fixture-only sealed artifact")
    generation = {"id": generation_id, "manifest_sha256": bench.sha256_file(manifest)}
    pointer = (f"FSFS-COMPLETE-GENERATION-v{version}\n{generation_id}\n"
               f'{generation["manifest_sha256"]}\n')
    if version == 2:
        pointer += "d" * 64 + "\n"
    (index / "FSFS-CURRENT").write_text(pointer, encoding="utf-8")
    return generation


class FixtureRunner:
    """In-memory protocol stand-in, intentionally incapable of writing a result receipt."""

    def __init__(self, output, mode, layout="legacy"):
        self.output = output
        self.mode = mode
        self.layout = layout
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
            index = Path(command[command.index("--index-dir") + 1])
            index.mkdir()
            if self.layout == "complete":
                complete_generation_fixture(index)
        elif (label.endswith("-cold") and self.layout == "legacy"
              or self.layout == "complete" and label.startswith("candidate-")
              and label.endswith(("-cold", "-semantic-control"))):
            index = Path(command[command.index("--index-dir") + 1])
            (index / "query_cache").mkdir()
            record = {"fixture_only": True}
            if self.layout == "complete":
                record["complete_generation"] = bench.complete_generation_snapshot(index)["generation"]
            bench.write_json(index / "query_cache" / "fixture-only.json", record)
        if "--filter" in command:
            response["data"].update(hits=[], returned_hits=0)
            if (self.layout == "legacy" and label.startswith("baseline-")
                    and label.endswith("-warm")):
                index = Path(command[command.index("--index-dir") + 1])
                path = index / "query_cache" / "fixture-only.json"
                stamp = path.stat()
                os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
        return {
            "response": response, "label": label, "elapsed_ms": 1,
            "model_loads": int(label.endswith(("-cold", "-semantic-control"))
                               or label.startswith("baseline-")),
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
        self.assertEqual(args.scenario, "semantic-hits")
        self.assertEqual(args.layout, "legacy")
        self.assertEqual(bench.parse_args(self.arguments(layout="complete")).layout, "complete")
        for option in (["--rounds", "0"], ["--timeout-seconds", "nan"],
                       ["--receipt-mode", "unknown"], ["--scenario", "unknown"],
                       ["--layout", "unknown"]):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.parse_args(self.arguments() + option)

    def test_selected_layout_replaces_inherited_runtime_overrides(self):
        for layout in bench.LAYOUTS:
            output = self.root / layout
            output.mkdir()
            with mock.patch.dict(os.environ, {
                "FSFS_COMPLETE_GENERATIONS": "inherited", "FSFS_TEST_ONLY": "inherited",
                "FRANKENSEARCH_MODEL_DIR": "inherited", "FRANKENSEARCH_OFFLINE": "0",
                "HF_HOME": "inherited", "HUGGINGFACE_HUB_CACHE": "inherited",
            }):
                environment = bench.runtime_environment(output, layout)
            self.assertEqual(environment["FSFS_COMPLETE_GENERATIONS"],
                             str(int(layout == "complete")))
            self.assertEqual(environment["FRANKENSEARCH_OFFLINE"], "1")
            self.assertEqual(environment["FRANKENSEARCH_ALLOW_DOWNLOAD"], "0")
            self.assertEqual(environment["FRANKENSEARCH_MODEL_DIR"], str(output / "models"))
            for key in ("FSFS_TEST_ONLY", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
                self.assertNotIn(key, environment)

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

    def empty_pair(self, arm="candidate"):
        cold, warm, before, after = self.pair(arm)
        for result in (cold, warm):
            result["response"]["data"].update(hits=[], returned_hits=0)
        if arm == "baseline":
            after["mtime_ns"] += 1
        return cold, warm, before, after

    def test_filtered_empty_has_distinct_strict_replay_and_identity_gates(self):
        for arm in bench.ARMS:
            cold, warm, before, after = self.empty_pair(arm)
            self.assertEqual(bench.validate_pair(
                arm, cold, warm, before, after, None, "valid",
                "filtered-empty", bench.MODEL_ID), [])
            with self.assertRaises(bench.BenchmarkError):
                bench.validate_pair(arm, cold, warm, before, after, None, "valid")
            with self.assertRaisesRegex(bench.BenchmarkError, "semantic control"):
                bench.validate_pair(arm, cold, warm, before, after, None, "valid",
                                    "filtered-empty")
            after["mtime_ns"] = before["mtime_ns"] + int(arm == "candidate")
            with self.assertRaisesRegex(bench.BenchmarkError, "rewrite|rewritten"):
                bench.validate_pair(arm, cold, warm, before, after, None, "valid",
                                    "filtered-empty", bench.MODEL_ID)
        mutations = [
            {"hits": [{"semantic_rank": 0}]}, {"returned_hits": 1},
            {"returned_hits": False}, {"phase": "refined"}, {"query": ""},
            {"vector_generation_id": "foreign-producer"}, {"vector_generation_is_hash": True},
        ]
        for changes in mutations:
            cold, warm, before, after = self.empty_pair()
            for result in (cold, warm):
                result["response"]["data"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(bench.BenchmarkError):
                bench.validate_pair("candidate", cold, warm, before, after, None, "valid",
                                    "filtered-empty", bench.MODEL_ID)
        for field, replacement in (("bytes", b"rewritten"), ("names", ["other.json"])):
            cold, warm, before, after = self.empty_pair("baseline")
            after[field] = replacement
            with self.subTest(field=field), self.assertRaises(bench.BenchmarkError):
                bench.validate_pair("baseline", cold, warm, before, after, None, "valid",
                                    "filtered-empty", bench.MODEL_ID)

    def test_filtered_empty_requires_controls_on_separate_indexes_in_every_receipt_mode(self):
        for mode in ("valid", "absent", "malformed"):
            with self.subTest(mode=mode):
                output = self.root / mode
                (output / "models" / bench.MODEL_DIRECTORY).mkdir(parents=True)
                runner = FixtureRunner(output, mode)
                binaries = {arm: self.root / (arm + "-fixture-only") for arm in bench.ARMS}
                template, config, state = bench.setup_fixture(runner, binaries, output, mode)
                with self.assertRaisesRegex(bench.BenchmarkError, "both semantic controls"):
                    bench.run_rounds(runner, binaries, output, template, config, 2, state,
                                     "filtered-empty")
                controls = bench.run_semantic_controls(
                    runner, binaries, output, template, config, state)
                rows = bench.run_rounds(runner, binaries, output, template, config, 2, state,
                                        "filtered-empty", controls)
                self.assertEqual([label for _, label, _ in runner.calls], [
                    "setup-verify-model", "setup-index",
                    "baseline-semantic-control", "candidate-semantic-control",
                    "baseline-0-cold", "baseline-0-warm", "candidate-0-cold", "candidate-0-warm",
                    "candidate-1-cold", "candidate-1-warm", "baseline-1-cold", "baseline-1-warm",
                ])
                self.assertEqual(len(rows), 8)
                self.assertFalse((template / "query_cache").exists())
                for arm in bench.ARMS:
                    self.assertTrue(controls[arm]["response"]["data"]["hits"])
                    self.assertEqual(controls[arm]["model_loads"], 1)
                for _, label, command in runner.calls[2:]:
                    index = Path(command[command.index("--index-dir") + 1])
                    self.assertNotEqual(index, template)
                    if label.endswith("-semantic-control"):
                        self.assertNotIn("--filter", command)
                        self.assertTrue(index.name.endswith("-semantic-control-index"))
                    else:
                        self.assertEqual(command[command.index("--filter") + 1], bench.EMPTY_FILTER)
                        self.assertFalse(index.name.endswith("-semantic-control-index"))
                for row in rows:
                    self.assertEqual(row["scenario"], "filtered-empty")
                    self.assertEqual(row["cache_mtime_unchanged"], row["arm"] == "candidate")
                    self.assertTrue(row["warm_payload_identical"] and row["cache_bytes_unchanged"])
                bench.summarize(rows, 2)

    def test_empty_or_incompatible_semantic_controls_stop_before_measurement(self):
        cases = [
            ("baseline", {"hits": []}, 1),
            ("baseline", {"hits": [{"semantic_rank": None}]}, 1),
            ("baseline", {"vector_generation_is_hash": True}, 1),
            ("baseline", {"vector_generation_id": ""}, 1),
            ("baseline", {"query": "wrong query"}, 1),
            ("baseline", {"phase": "refined"}, 1),
            ("baseline", {"returned_hits": 0}, 1),
            ("baseline", {}, 0),
            ("candidate", {"vector_generation_id": "foreign-producer"}, 1),
            ("candidate", {"hits": [{"semantic_rank": 0, "score": 9}]}, 1),
        ]
        for ordinal, (arm, changes, loads) in enumerate(cases):
            output = self.root / f"control-{ordinal}"
            (output / "models" / bench.MODEL_DIRECTORY).mkdir(parents=True)
            runner = FixtureRunner(output, "valid")
            binaries = {name: self.root / (name + "-fixture-only") for name in bench.ARMS}
            template, config, state = bench.setup_fixture(runner, binaries, output, "valid")
            invoke = runner.invoke

            def corrupt_control(binary, label, command):
                result = invoke(binary, label, command)
                if label == f"{arm}-semantic-control":
                    result["response"]["data"].update(changes)
                    result["model_loads"] = loads
                return result

            with mock.patch.object(runner, "invoke", side_effect=corrupt_control):
                with self.subTest(case=ordinal), self.assertRaises(bench.BenchmarkError):
                    bench.run_semantic_controls(runner, binaries, output, template, config, state)
            self.assertFalse(any(label.endswith(("-cold", "-warm"))
                                 for _, label, _ in runner.calls))

    def test_filtered_empty_rejects_complete_payload_drift_across_arms_and_rounds(self):
        for ordinal, prefix in enumerate(("candidate-0-", "candidate-1-")):
            output = self.root / f"payload-drift-{ordinal}"
            (output / "models" / bench.MODEL_DIRECTORY).mkdir(parents=True)
            runner = FixtureRunner(output, "valid")
            binaries = {arm: self.root / (arm + "-fixture-only") for arm in bench.ARMS}
            template, config, state = bench.setup_fixture(runner, binaries, output, "valid")
            controls = bench.run_semantic_controls(
                runner, binaries, output, template, config, state)
            invoke = runner.invoke

            def corrupt_payload(binary, label, command):
                result = invoke(binary, label, command)
                if label.startswith(prefix):
                    result["response"]["data"]["skip_reason"] = "fixture-only policy drift"
                return result

            with mock.patch.object(runner, "invoke", side_effect=corrupt_payload):
                with self.subTest(prefix=prefix), self.assertRaisesRegex(
                        bench.BenchmarkError, "payload differs across arms or rounds"):
                    bench.run_rounds(runner, binaries, output, template, config, 2, state,
                                     "filtered-empty", controls)

    def test_existing_answer_cannot_be_measured_as_cold(self):
        (self.root / "models" / bench.MODEL_DIRECTORY).mkdir(parents=True)
        runner = FixtureRunner(self.root, "valid")
        binaries = {arm: self.root / (arm + "-fixture-only") for arm in bench.ARMS}
        template, config, state = bench.setup_fixture(runner, binaries, self.root, "valid")
        (template / "query_cache").mkdir()
        (template / "query_cache" / "existing.json").write_bytes(b"fixture-only")
        with self.assertRaisesRegex(bench.BenchmarkError, "already has a persisted answer"):
            bench.run_rounds(runner, binaries, self.root, template, config, 1, state)
        self.assertEqual([label for _, label, _ in runner.calls],
                         ["setup-verify-model", "setup-index"])

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

    def test_complete_layout_keeps_baseline_uncached_and_candidate_bound_to_sealed_bundle(self):
        for scenario in bench.SCENARIOS:
            for mode in ("valid", "absent", "malformed"):
                with self.subTest(scenario=scenario, mode=mode):
                    output = self.root / f"{scenario}-{mode}"
                    (output / "models" / bench.MODEL_DIRECTORY).mkdir(parents=True)
                    runner = FixtureRunner(output, mode, "complete")
                    binaries = {arm: self.root / (arm + "-fixture-only") for arm in bench.ARMS}
                    template, config, state = bench.setup_fixture(
                        runner, binaries, output, mode, "complete")
                    original = bench.complete_generation_snapshot(template)
                    controls = bench.run_semantic_controls(
                        runner, binaries, output, template, config, state, "complete"
                    ) if scenario == "filtered-empty" else None
                    rows = bench.run_rounds(
                        runner, binaries, output, template, config, 2, state,
                        scenario, controls, "complete")
                    self.assertEqual(len(rows), 8)
                    self.assertEqual(bench.complete_generation_snapshot(template), original)
                    self.assertFalse((template / "query_cache").exists())
                    expected_order = [
                        "baseline-0-cold", "baseline-0-warm", "candidate-0-cold", "candidate-0-warm",
                        "candidate-1-cold", "candidate-1-warm", "baseline-1-cold", "baseline-1-warm",
                    ]
                    self.assertEqual([label for _, label, _ in runner.calls[-8:]], expected_order)
                    for row in rows:
                        cached = row["arm"] == "candidate"
                        self.assertEqual(row["layout"], "complete")
                        self.assertEqual(row["scenario"], scenario)
                        self.assertEqual(row["cache_record_count"], int(cached))
                        self.assertEqual(row["cache_sha256"] is not None, cached)
                        self.assertEqual(row["model_loads"],
                                         int(row["phase"] == "cold" or not cached))
                        self.assertEqual(row["complete_generation"], original["generation"])
                        self.assertEqual(row["sealed_bundle_inventory"], original["inventory"])
                        self.assertEqual(row["selection_pointer_sha256"], original["pointer_sha256"])
                        for field in ("warm_payload_identical", "cache_bytes_unchanged",
                                      "cache_mtime_unchanged", "sealed_bundle_unchanged"):
                            self.assertTrue(row[field])
                    if controls is not None:
                        for arm in bench.ARMS:
                            self.assertTrue(controls[arm]["response"]["data"]["hits"])
                            self.assertEqual(controls[arm]["model_loads"], 1)
                            self.assertEqual(controls[arm]["complete_generation"], original["generation"])
                            control_index = output / f"{arm}-semantic-control-index"
                            self.assertEqual(len(list((control_index / "query_cache").glob("*.json"))),
                                             int(arm == "candidate"))
                    bench.summarize(rows, 2)

    def test_complete_snapshot_requires_valid_pointer_framing_and_manifest_digest(self):
        for version in (1, 2):
            index = self.root / f"version-{version}"
            generation = complete_generation_fixture(index, version)
            self.assertEqual(bench.complete_generation_snapshot(index)["generation"], generation)
        malformed = [
            b"not-a-pointer", b"\xff", b"x" * 257,
            b"FSFS-COMPLETE-GENERATION-v1\n../escape\n" + b"a" * 64 + b"\n",
        ]
        for ordinal, contents in enumerate(malformed):
            index = self.root / f"malformed-{ordinal}"
            complete_generation_fixture(index)
            (index / "FSFS-CURRENT").write_bytes(contents)
            with self.subTest(case=ordinal), self.assertRaises(bench.BenchmarkError):
                bench.complete_generation_snapshot(index)
        index = self.root / "changed-manifest"
        generation = complete_generation_fixture(index)
        (index / "generations" / generation["id"] / "FSFS-BUNDLE.json").write_bytes(b"changed")
        with self.assertRaisesRegex(bench.BenchmarkError, "manifest digest differs"):
            bench.complete_generation_snapshot(index)
        index = self.root / "missing-generation"
        generation = complete_generation_fixture(index)
        bundle = index / "generations" / generation["id"]
        bundle.rename(bundle.with_name("moved-aside"))
        with self.assertRaisesRegex(bench.BenchmarkError, "missing or indirect"):
            bench.complete_generation_snapshot(index)

    def test_complete_layout_refuses_cache_and_bundle_drift_during_either_request(self):
        cases = [
            ("baseline-cache", "baseline-0-cold", "expected 0 persisted"),
            ("missing-cache", "candidate-0-cold", "expected 1 persisted"),
            ("extra-cache", "candidate-0-warm", "expected 1 persisted"),
            ("missing-stamp", "candidate-0-cold", "exact selected complete generation"),
            ("wrong-id", "candidate-0-warm", "exact selected complete generation"),
            ("wrong-digest", "candidate-0-cold", "exact selected complete generation"),
            ("invalid-cache", "candidate-0-warm", "invalid complete-generation cache"),
            ("bundle-bytes", "baseline-0-cold", "sealed bundle changed"),
            ("bundle-rewrite", "candidate-0-warm", "sealed bundle changed"),
            ("pointer-change", "baseline-0-warm", "sealed bundle changed"),
            ("bundled-cache", "candidate-0-cold", "outside the sealed"),
            ("cache-rewrite", "candidate-0-warm", "warm cache was rewritten"),
        ]
        for case, trigger, error in cases:
            output = self.root / case
            (output / "models" / bench.MODEL_DIRECTORY).mkdir(parents=True)
            runner = FixtureRunner(output, "valid", "complete")
            binaries = {arm: self.root / (arm + "-fixture-only") for arm in bench.ARMS}
            template, config, state = bench.setup_fixture(
                runner, binaries, output, "valid", "complete")
            invoke = runner.invoke

            def corrupt_search(binary, label, command):
                result = invoke(binary, label, command)
                if label != trigger:
                    return result
                index = Path(command[command.index("--index-dir") + 1])
                generation = bench.complete_generation_snapshot(index)["generation"]
                bundle = index / "generations" / generation["id"]
                cache = index / "query_cache" / "fixture-only.json"
                if case == "baseline-cache":
                    cache.parent.mkdir()
                    cache.write_bytes(b"unexpected baseline answer")
                elif case == "missing-cache":
                    cache.rename(index / "moved-answer.json")
                elif case == "extra-cache":
                    (cache.parent / "extra.json").write_bytes(cache.read_bytes())
                elif case in ("missing-stamp", "wrong-id", "wrong-digest"):
                    record = json.loads(cache.read_bytes())
                    if case == "missing-stamp":
                        record.pop("complete_generation")
                    else:
                        field = "id" if case == "wrong-id" else "manifest_sha256"
                        record["complete_generation"][field] = "foreign-generation"
                    bench.write_json(cache, record)
                elif case == "invalid-cache":
                    cache.write_bytes(b"not-json")
                elif case == "bundle-bytes":
                    (bundle / "vectors.fixture").write_bytes(b"changed sealed artifact")
                elif case == "bundle-rewrite":
                    path = bundle / "vectors.fixture"
                    stamp = path.stat()
                    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
                elif case == "pointer-change":
                    path = index / "FSFS-CURRENT"
                    path.write_bytes(path.read_bytes().replace(b"d" * 64, b"e" * 64))
                elif case == "bundled-cache":
                    (bundle / "query_cache").mkdir()
                elif case == "cache-rewrite":
                    stamp = cache.stat()
                    os.utime(cache, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
                return result

            with mock.patch.object(runner, "invoke", side_effect=corrupt_search):
                with self.subTest(case=case), self.assertRaisesRegex(bench.BenchmarkError, error):
                    bench.run_rounds(runner, binaries, output, template, config, 1, state,
                                     layout="complete")

    def test_complete_semantic_control_enforces_generation_integrity_before_empty_measurement(self):
        for target in bench.ARMS:
            output = self.root / target
            (output / "models" / bench.MODEL_DIRECTORY).mkdir(parents=True)
            runner = FixtureRunner(output, "valid", "complete")
            binaries = {arm: self.root / (arm + "-fixture-only") for arm in bench.ARMS}
            template, config, state = bench.setup_fixture(
                runner, binaries, output, "valid", "complete")
            invoke = runner.invoke

            def corrupt_control(binary, label, command):
                result = invoke(binary, label, command)
                if label == f"{target}-semantic-control":
                    index = Path(command[command.index("--index-dir") + 1])
                    generation = bench.complete_generation_snapshot(index)["generation"]
                    (index / "generations" / generation["id"] / "new.fixture").write_bytes(b"changed")
                return result

            with mock.patch.object(runner, "invoke", side_effect=corrupt_control):
                with self.subTest(target=target), self.assertRaisesRegex(
                        bench.BenchmarkError, "sealed bundle changed"):
                    bench.run_semantic_controls(
                        runner, binaries, output, template, config, state, "complete")
            self.assertFalse(any(label.endswith(("-cold", "-warm"))
                                 for _, label, _ in runner.calls))

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
