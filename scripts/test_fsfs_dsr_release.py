#!/usr/bin/env python3
"""DSR validator regressions with fake binaries and a mocked verifier boundary.

These tests establish inventory/integrity orchestration, not cryptographic or
native execution evidence. The dsr-release CLI exposes no mock-verifier option.
Fixture directories are retained for failure inspection.
"""

import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

import check_fsfs_dsr_release as check


VERSION = "1.13.0"
SOURCE = "a" * 40
FIXTURE_SIGNATURE = b"orchestration fixture, not a cryptographic signature\n"
LITE_STEM = "fsfs-lite-1.13.0-x86_64-unknown-linux-musl"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fake_binary(target):
    header = bytearray(64)
    if "linux" in target:
        header[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", header, 16, 2)
        struct.pack_into("<H", header, 18, 183 if target.startswith("aarch64") else 62)
    else:
        header[:4] = b"\xcf\xfa\xed\xfe"
        struct.pack_into("<I", header, 4, 0x0100000C if target.startswith("aarch64") else 0x01000007)
        struct.pack_into("<I", header, 12, 2)
    return bytes(header) + b"NOT A RUNNABLE BINARY: " + target.encode()


def tar_bytes(binary, member_names=("fsfs",), link=False, mode=0o755):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:xz", preset=0) as archive:
        for name in member_names:
            member = tarfile.TarInfo(name)
            member.mode = mode
            member.uid = member.gid = 1000
            member.size = len(binary)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside"
                member.size = 0
            archive.addfile(member, None if link else io.BytesIO(binary))
        for name in ("README.md", "LICENSE"):
            member = tarfile.TarInfo(name)
            member.mode = 0o664
            data = b"fixture documentation\n"
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def refresh_checksums(files):
    payloads, _ = check.payloads_and_aliases(VERSION)
    for name in payloads:
        files[name + ".sha256"] = f"{digest(files[name])}  {name}\n".encode()
    files["SHA256SUMS"] = b"".join(
        f"{digest(files[name])}  {name}\n".encode() for name in sorted(payloads)
    )


def fixture_files():
    files = {}
    for (profile, target), stem in check.stems(VERSION).items():
        binary = fake_binary(target)
        files[stem + ".tar.xz"] = tar_bytes(binary)
        files[stem + ".metadata.json"] = json.dumps({
            "tag": "v" + VERSION, "target": target, "binary": "fsfs", "profile": profile,
            "source_revision": SOURCE, "build_timestamp_utc": "2026-11-04T00:00:00Z",
            "rustc": "rustc fixture", "semantic_loaders": profile == "default",
            "embedded_models": False, "binary_sha256": "sha256:" + digest(binary),
        }).encode()
    payloads, aliases = check.payloads_and_aliases(VERSION)
    for alias, original in aliases.items():
        files[alias] = files[original]
    for name in payloads | {"SHA256SUMS"}:
        files[name + ".minisig"] = FIXTURE_SIGNATURE
    refresh_checksums(files)
    return files


class FixtureVerifier:
    """In-process test double. This class is never used by dsr-release."""
    calls = []

    def __init__(self, _executable):
        pass

    def verify(self, message):
        self.calls.append(message.name)
        check.require(message.with_name(message.name + ".minisig").read_bytes() == FIXTURE_SIGNATURE,
                      "signature_invalid", message.name)


class DsrReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact_root = Path(tempfile.mkdtemp(prefix="fsfs-dsr-contract-"))
        print(f"Fixture artifacts retained at {cls.artifact_root}", flush=True)

    def setUp(self):
        self.files = fixture_files()
        self.directory = self.artifact_root / self._testMethodName
        self.directory.mkdir()
        FixtureVerifier.calls = []

    def materialize(self):
        for name, data in self.files.items():
            (self.directory / name).write_bytes(data)

    def validate(self):
        self.materialize()
        with mock.patch.object(check, "MinisignVerifier", FixtureVerifier), mock.patch.object(
            check.subprocess, "run", side_effect=AssertionError("fixture must not execute processes")
        ):
            return check.validate_release(self.directory, VERSION, SOURCE, "unused-test-double")

    def rejected(self, code):
        with self.assertRaises(check.ReleaseError) as caught:
            self.validate()
        self.assertEqual(caught.exception.code, code)

    def change_metadata(self, key, value):
        path = LITE_STEM + ".metadata.json"
        metadata = json.loads(self.files[path])
        metadata[key] = value
        self.files[path] = json.dumps(metadata).encode()
        refresh_checksums(self.files)

    def test_positive_exact_inventory_and_no_execution_claim(self):
        result = self.validate()
        self.assertEqual(result["artifact_count"], 50)
        self.assertEqual(len(result["binaries"]), 6)
        self.assertEqual(len(FixtureVerifier.calls), 17)
        self.assertEqual(len(set(FixtureVerifier.calls)), 17)
        self.assertEqual(result["native_execution"], "NOT_PERFORMED")
        self.assertEqual(result["proxy_execution"], "NOT_PERFORMED")
        self.assertEqual(result["unfulfilled_historical_platform_obligations"],
                         ["Full Windows", "Full Linux ARM GNU"])

    def test_missing_asset(self):
        self.files.pop(LITE_STEM + ".tar.xz.minisig")
        self.rejected("inventory")

    def test_extra_asset(self):
        self.files["unreviewed-binary"] = b"extra"
        self.rejected("inventory")

    def test_duplicate_checksum_entry(self):
        self.files["SHA256SUMS"] += self.files["SHA256SUMS"].splitlines(keepends=True)[0]
        self.rejected("duplicate_checksum")

    def test_duplicate_metadata_key(self):
        name = LITE_STEM + ".metadata.json"
        self.files[name] = self.files[name][:-1] + b', "profile": "lite"}'
        refresh_checksums(self.files)
        self.rejected("duplicate_metadata_key")

    def test_checksum_mismatch(self):
        self.files[LITE_STEM + ".tar.xz"] += b"changed"
        self.rejected("checksum_mismatch")

    def test_wrong_sidecar_filename(self):
        name = LITE_STEM + ".tar.xz"
        self.files[name + ".sha256"] = f"{digest(self.files[name])}  other.tar.xz\n".encode()
        self.rejected("checksum_inventory")

    def test_missing_manifest_entry(self):
        self.files["SHA256SUMS"] = b"".join(self.files["SHA256SUMS"].splitlines(keepends=True)[1:])
        self.rejected("checksum_inventory")

    def test_signature_rejection(self):
        self.files[LITE_STEM + ".metadata.json.minisig"] = b"invalid fixture signature"
        self.rejected("signature_invalid")

    def test_wrong_source(self):
        self.change_metadata("source_revision", "b" * 40)
        self.rejected("metadata_field")

    def test_wrong_tag(self):
        self.change_metadata("tag", "v1.12.3")
        self.rejected("metadata_field")

    def test_wrong_profile(self):
        self.change_metadata("profile", "default")
        self.rejected("metadata_field")

    def test_false_loader_capability(self):
        self.change_metadata("semantic_loaders", True)
        self.rejected("metadata_field")

    def test_nonboolean_capability(self):
        self.change_metadata("embedded_models", 0)
        self.rejected("metadata_field")

    def test_binary_digest_mismatch(self):
        self.change_metadata("binary_sha256", "sha256:" + "0" * 64)
        self.rejected("binary_checksum")

    def test_wrong_binary_architecture(self):
        binary = fake_binary("aarch64-unknown-linux-musl")
        self.files[LITE_STEM + ".tar.xz"] = tar_bytes(binary)
        self.change_metadata("binary_sha256", "sha256:" + digest(binary))
        self.rejected("binary_architecture")

    def test_relocatable_elf_object_rejected(self):
        binary = bytearray(fake_binary("x86_64-unknown-linux-musl"))
        struct.pack_into("<H", binary, 16, 1)
        self.files[LITE_STEM + ".tar.xz"] = tar_bytes(bytes(binary))
        self.change_metadata("binary_sha256", "sha256:" + digest(binary))
        self.rejected("binary_format")

    def test_archive_binary_requires_executable_mode(self):
        self.files[LITE_STEM + ".tar.xz"] = tar_bytes(fake_binary("x86_64-unknown-linux-musl"), mode=0o644)
        refresh_checksums(self.files)
        self.rejected("binary_not_executable")

    def test_alias_divergence_even_with_consistent_checksums(self):
        self.files["fsfs-aarch64-apple-darwin.metadata.json"] += b"\n"
        refresh_checksums(self.files)
        self.rejected("alias_mismatch")

    def test_alias_checksum_is_not_copied_from_versioned_name(self):
        alias = "fsfs-aarch64-apple-darwin.tar.xz"
        self.files[alias + ".sha256"] = self.files["fsfs-1.13.0-aarch64-apple-darwin.tar.xz.sha256"]
        self.rejected("checksum_inventory")

    def test_traversal_archive_member(self):
        self.files[LITE_STEM + ".tar.xz"] = tar_bytes(fake_binary("x86_64-unknown-linux-musl"), ("../fsfs",))
        refresh_checksums(self.files)
        self.rejected("unsafe_archive_path")

    def test_duplicate_archive_binary(self):
        self.files[LITE_STEM + ".tar.xz"] = tar_bytes(fake_binary("x86_64-unknown-linux-musl"), ("fsfs", "./fsfs"))
        refresh_checksums(self.files)
        self.rejected("duplicate_archive_member")

    def test_symlink_archive_binary(self):
        self.files[LITE_STEM + ".tar.xz"] = tar_bytes(b"", link=True)
        refresh_checksums(self.files)
        self.rejected("unsafe_archive_member")

    def test_asset_directory_rejected(self):
        name = LITE_STEM + ".tar.xz"
        self.files.pop(name)
        (self.directory / name).mkdir()
        self.rejected("nonregular_asset")

    def test_asset_symlink_rejected(self):
        name = LITE_STEM + ".tar.xz"
        self.files.pop(name)
        (self.directory / name).symlink_to("SHA256SUMS")
        self.rejected("nonregular_asset")

    def test_missing_verifier(self):
        with mock.patch.object(check.shutil, "which", return_value=None):
            with self.assertRaises(check.ReleaseError) as caught:
                check.MinisignVerifier("not-installed")
        self.assertEqual(caught.exception.code, "verifier_missing")

    def test_accept_everything_verifier_rejected(self):
        version = subprocess.CompletedProcess([], 0, b"minisign 0.12\n", b"")
        accepted = subprocess.CompletedProcess([], 0, b"", b"")
        with mock.patch.object(check.shutil, "which", return_value="fixture"), mock.patch.object(
            check.subprocess, "run", side_effect=[version, accepted, accepted]
        ):
            with self.assertRaises(check.ReleaseError) as caught:
                check.MinisignVerifier("fixture")
        self.assertEqual(caught.exception.code, "verifier_canary")

    def test_verifier_calls_pin_key_and_explicit_signature(self):
        version = subprocess.CompletedProcess([], 0, b"minisign 0.12\n", b"")
        accepted = subprocess.CompletedProcess([], 0, b"", b"")
        rejected = subprocess.CompletedProcess([], 1, b"", b"")
        with mock.patch.object(check.shutil, "which", return_value="fixture"), mock.patch.object(
            check.subprocess, "run", side_effect=[version, accepted, rejected, accepted]
        ) as runner:
            verifier = check.MinisignVerifier("fixture")
            verifier.verify(Path("/artifact/fsfs.tar.xz"))
        self.assertEqual(runner.call_args_list[-1].args[0], [
            "fixture", "-V", "-m", "/artifact/fsfs.tar.xz", "-x", "/artifact/fsfs.tar.xz.minisig",
            "-P", check.PUBLIC_KEY,
        ])


if __name__ == "__main__":
    print("DSR validator fixtures use mocked signatures and non-runnable binaries; no release qualification is claimed.", flush=True)
    unittest.main(verbosity=2)
