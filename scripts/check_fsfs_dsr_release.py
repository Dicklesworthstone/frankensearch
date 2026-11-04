#!/usr/bin/env python3
"""Check the existing DSR release format without publishing or executing fsfs.

This is called by check_fsfs_packaging_release_install_contract.sh. The older
schema fixtures retain their separate platform obligations. Passing here proves
signed staged-file integrity and metadata consistency, not native execution,
proxy behavior, a reproducible build, or completion of the release quality gate.
"""

import argparse
import datetime
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parent.parent
PUBLIC_KEY = "RWTQGPeLsnm9G7VFdFWkkcRi3wJK/PqsYxWC+oLNN74W9IjBxRU1Xu70"
VARIANTS = (
    ("default", "x86_64-unknown-linux-gnu"),
    ("default", "aarch64-apple-darwin"),
    ("lite", "x86_64-unknown-linux-musl"),
    ("lite", "aarch64-unknown-linux-musl"),
    ("lite", "x86_64-apple-darwin"),
    ("lite", "aarch64-apple-darwin"),
)
MAX_TEXT_BYTES = 64 * 1024
MAX_BINARY_BYTES = 2 * 1024**3


class ReleaseError(Exception):
    def __init__(self, code, detail):
        super().__init__(f"release.dsr.{code}: {detail}")
        self.code = code


def require(condition, code, detail):
    if not condition:
        raise ReleaseError(code, detail)


def version_and_source(version, source):
    version = version.removeprefix("v")
    require(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", version),
            "invalid_version", "expected a release version, for example 1.13.0")
    require(re.fullmatch(r"[0-9a-f]{40}", source), "invalid_source",
            "source revision must be the full lowercase Git commit SHA")
    return version


def stems(version):
    return {
        (profile, target): f"fsfs-{'lite-' if profile == 'lite' else ''}{version}-{target}"
        for profile, target in VARIANTS
    }


def payloads_and_aliases(version):
    payloads = set()
    aliases = {}
    for (profile, target), stem in stems(version).items():
        for suffix in (".tar.xz", ".metadata.json"):
            name = stem + suffix
            payloads.add(name)
            if profile == "default":
                alias = f"fsfs-{target}{suffix}"
                payloads.add(alias)
                aliases[alias] = name
    return payloads, aliases


def expected_inventory(version):
    payloads, _ = payloads_and_aliases(version)
    return {name + suffix for name in payloads for suffix in ("", ".sha256", ".minisig")} | {
        "SHA256SUMS", "SHA256SUMS.minisig"
    }


def inventory(directory, expected):
    require(directory.is_dir(), "artifact_directory", f"not a directory: {directory}")
    entries = {path.name: path for path in directory.iterdir()}
    require(set(entries) == expected, "inventory",
            f"missing={sorted(expected - set(entries))}, extra={sorted(set(entries) - expected)}")
    for name, path in entries.items():
        require(stat.S_ISREG(path.lstat().st_mode), "nonregular_asset",
                f"expected a regular file, not a link or directory: {name}")
        require(path.stat().st_size > 0, "empty_asset", name)
    return entries


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path):
    require(path.stat().st_size <= MAX_TEXT_BYTES, "oversized_text", path.name)
    return path.read_text(encoding="utf-8")


def checksums(path, expected):
    result = {}
    for line in read_text(path).splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *]([^/\\\r\n]+)", line)
        require(match is not None, "malformed_checksum", path.name)
        digest, name = match.groups()
        require(name not in result, "duplicate_checksum", f"{path.name}: {name}")
        result[name] = digest.lower()
    require(set(result) == expected, "checksum_inventory",
            f"{path.name}: missing={sorted(expected - set(result))}, extra={sorted(set(result) - expected)}")
    return result


class MinisignVerifier:
    """Use only the pinned repository trust key; no test bypass is exposed."""

    def __init__(self, executable):
        self.executable = shutil.which(executable)
        require(self.executable is not None, "verifier_missing", executable)
        key_lines = (ROOT / "minisign.pub").read_text(encoding="utf-8").splitlines()
        require(len(key_lines) == 2 and key_lines[1] == PUBLIC_KEY,
                "trust_key_mismatch", "installer/release trust key requires deliberate review")
        version = subprocess.run([self.executable, "-v"], capture_output=True, timeout=30)
        require(version.returncode == 0 and re.fullmatch(
            rb"minisign [0-9]+\.[0-9]+(?:\.[0-9]+)?\s*", version.stdout),
            "verifier_identity", "the executable must identify itself as Minisign")
        # Public, authenticated v1.12.3 metadata is a verifier canary, never
        # evidence that the candidate passed native execution. Verify it through
        # stdin first so the tamper rejection cannot merely be a stdin failure.
        # Original asset: releases/download/v1.12.3/
        # fsfs-1.12.3-aarch64-apple-darwin.metadata.json in this GitHub repository.
        canary = ROOT / "schemas/fixtures/fsfs-dsr-minisign-canary.metadata.json"
        signature = canary.with_name(canary.name + ".minisig")
        message = canary.read_bytes()
        require(self._run(Path("/dev/stdin"), signature, message).returncode == 0,
                "verifier_canary", "authentic pinned-key canary was rejected")
        require(self._run(Path("/dev/stdin"), signature, message + b"\n").returncode != 0,
                "verifier_canary", "verifier accepted the tampered canary")

    def _run(self, message, signature, data=None):
        return subprocess.run(
            [self.executable, "-V", "-m", str(message), "-x", str(signature), "-P", PUBLIC_KEY],
            input=data, capture_output=True, timeout=120,
        )

    def verify(self, message):
        signature = message.with_name(message.name + ".minisig")
        require(self._run(message, signature).returncode == 0,
                "signature_invalid", message.name)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_metadata_key", key)
        result[key] = value
    return result


def metadata(path, version, source, profile, target):
    data = json.loads(read_text(path), object_pairs_hook=unique_object)
    require(isinstance(data, dict), "metadata_object", path.name)
    expected = {
        "tag": f"v{version}", "target": target, "binary": "fsfs",
        "profile": profile, "source_revision": source,
        "semantic_loaders": profile == "default", "embedded_models": False,
    }
    for key, value in expected.items():
        require(type(data.get(key)) is type(value) and data[key] == value,
                "metadata_field", f"{path.name}: {key} must equal {value!r}")
    require(isinstance(data.get("rustc"), str) and data["rustc"].startswith("rustc "),
            "metadata_rustc", path.name)
    timestamp = data.get("build_timestamp_utc")
    require(isinstance(timestamp, str), "metadata_timestamp", path.name)
    try:
        parsed_time = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReleaseError("metadata_timestamp", path.name) from error
    require(parsed_time.utcoffset() == datetime.timedelta(0), "metadata_timestamp", path.name)
    binary_digest = data.get("binary_sha256")
    require(isinstance(binary_digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", binary_digest),
            "metadata_binary_digest", path.name)
    return binary_digest.removeprefix("sha256:")


def binary_architecture(header, target, name):
    if "linux" in target:
        require(len(header) >= 64 and header[:6] == b"\x7fELF\x02\x01",
                "binary_format", f"{name}: expected a 64-bit little-endian ELF")
        require(struct.unpack_from("<H", header, 16)[0] in (2, 3),
                "binary_format", f"{name}: ELF is neither an executable nor a position-independent executable")
        machine = struct.unpack_from("<H", header, 18)[0]
        expected_machine = 183 if target.startswith("aarch64") else 62
    else:
        require(len(header) >= 32 and header[:4] == b"\xcf\xfa\xed\xfe",
                "binary_format", f"{name}: expected a 64-bit little-endian Mach-O")
        machine = struct.unpack_from("<I", header, 4)[0]
        expected_machine = 0x0100000C if target.startswith("aarch64") else 0x01000007
        require(struct.unpack_from("<I", header, 12)[0] == 2,
                "binary_format", f"{name}: Mach-O is not an executable")
    require(machine == expected_machine, "binary_architecture", f"{name}: wrong target machine")


def archive_binary(path, target, expected_digest):
    """Stream and hash regular entries; never extract or execute archive content."""
    seen = set()
    binary_digest = None
    with tarfile.open(path, mode="r|xz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            require(not member.name.startswith("/") and ".." not in parts and "\\" not in member.name,
                    "unsafe_archive_path", f"{path.name}: {member.name!r}")
            name = str(PurePosixPath(member.name))
            require(name not in seen, "duplicate_archive_member", f"{path.name}: {name}")
            seen.add(name)
            if member.isdir() and name == ".":
                continue
            require(name in {"fsfs", "README.md", "LICENSE"} and member.isfile(),
                    "unsafe_archive_member", f"{path.name}: {name!r}")
            require(not member.mode & 0o6000, "unsafe_archive_mode", f"{path.name}: {name}")
            require(0 < member.size <= MAX_BINARY_BYTES, "archive_member_size", f"{path.name}: {name}")
            if name != "fsfs":
                continue
            require(member.mode & 0o100, "binary_not_executable", path.name)
            digest = hashlib.sha256()
            with archive.extractfile(member) as binary:
                header = binary.read(64)
                binary_architecture(header, target, path.name)
                digest.update(header)
                for chunk in iter(lambda: binary.read(1024 * 1024), b""):
                    digest.update(chunk)
            binary_digest = digest.hexdigest()
    require(binary_digest is not None, "binary_missing", path.name)
    require(binary_digest == expected_digest, "binary_checksum", path.name)
    return binary_digest


def validate_release(directory, version, source, minisign):
    version = version_and_source(version, source)
    directory = Path(directory).resolve()
    expected = expected_inventory(version)
    files = inventory(directory, expected)
    verifier = MinisignVerifier(minisign)
    verifier.verify(files["SHA256SUMS"])
    payloads, aliases = payloads_and_aliases(version)
    manifest = checksums(files["SHA256SUMS"], payloads)
    digests = {name: digest_file(path) for name, path in files.items()}
    for name in sorted(payloads):
        sidecar = checksums(files[name + ".sha256"], {name})
        require(manifest[name] == sidecar[name] == digests[name], "checksum_mismatch", name)
        verifier.verify(files[name])
    for alias, original in aliases.items():
        require(digests[alias] == digests[original], "alias_mismatch", f"{alias} != {original}")
    binaries = []
    for (profile, target), stem in stems(version).items():
        binary_digest = metadata(files[stem + ".metadata.json"], version, source, profile, target)
        archive_binary(files[stem + ".tar.xz"], target, binary_digest)
        binaries.append({"profile": profile, "target": target, "binary_sha256": binary_digest})
    # Refuse a staging directory changed while the checks ran. Upload must later
    # use these exact reviewed hashes; this check does not freeze a directory.
    final_files = inventory(directory, expected)
    require(all(digest_file(path) == digests[name] for name, path in final_files.items()),
            "artifacts_changed", "staged bytes changed during validation")
    return {
        "status": "STAGED_INTEGRITY_VERIFIED_NOT_RELEASE_QUALIFICATION",
        "tag": f"v{version}", "signed_metadata_source_revision": source,
        "artifact_count": len(files), "signatures_verified": len(payloads) + 1,
        "binaries": binaries,
        "artifacts": [{"name": name, "sha256": digests[name], "bytes": files[name].stat().st_size}
                      for name in sorted(files)],
        "native_execution": "NOT_PERFORMED", "proxy_execution": "NOT_PERFORMED",
        "unfulfilled_historical_platform_obligations": ["Full Windows", "Full Linux ARM GNU"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--minisign", default="minisign")
    args = parser.parse_args()
    try:
        result = validate_release(args.artifact_dir, args.version, args.source_revision, args.minisign)
    except (ReleaseError, OSError, ValueError, EOFError, lzma.LZMAError,
            tarfile.TarError, subprocess.SubprocessError) as error:
        print(f"{error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
