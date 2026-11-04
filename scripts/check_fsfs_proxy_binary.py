#!/usr/bin/env python3
"""Run bounded proxy wire/security probes against an explicit, hash-bound fsfs.

Called by the existing packaging checker. Local peers deliberately refuse the
registered model request or present an untrusted test certificate. A pass proves
the observed executable's proxy exchanges and refusal behavior, not provisioning,
independent hostname verification, or absence of direct Internet connections.

--provision explicitly enables live, authenticated byte-forwarding proxies and
checks the complete pinned 31 MB Potion bundle, plus a matching NO_PROXY replay
into another empty cache. Origin TLS remains end to end in fsfs. These results
cannot attribute every final byte to proxy traffic, attest direct-egress isolation,
or prove zero direct traffic. Host architecture alone does not exclude emulation.
No build, binary retrieval, trust override, release, or publication is performed.
All work directories and sanitized command output are retained, including failure.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import selectors
import signal
import socket
import ssl
import stat
import subprocess
import sys
import threading
import time


MODEL = "potion-base-8m"
MODEL_DIRECTORY = "potion-base-8M"
MODEL_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"
MODEL_FILES = {
    "tokenizer.json": (683_666, "e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50"),
    "model.safetensors": (30_236_760, "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2"),
}
ORIGIN = "huggingface.co"
MAX_OUTPUT_BYTES = 128 * 1024
MAX_OUTPUT_LINES = 1024
MAX_HEADER_BYTES = 8192
MAX_TUNNEL_BYTES = 64 * 1024 * 1024
MAX_PROXY_CONNECTIONS = 32
TLS_REFUSAL_ALERTS = {
    "TLSV1_ALERT_UNKNOWN_CA", "SSLV3_ALERT_BAD_CERTIFICATE",
    "SSLV3_ALERT_CERTIFICATE_UNKNOWN", "TLSV1_ALERT_CERTIFICATE_UNKNOWN",
}

# The public test-only origin certificate/key from embed/src/proxy.rs's
# tls_test_support. The private scalar is deliberately 3, valid 2020--2100.
# This is not a production credential. No root is added to the tested client.
TEST_CERTIFICATE = """-----BEGIN CERTIFICATE-----
MIIB3DCCAYOgAwIBAgIBAzAKBggqhkjOPQQDAjAwMS4wLAYDVQQDDCVGcmFua2Vu
U2VhcmNoIHByb3h5IHRyYW5zcG9ydCB0ZXN0IENBMCAXDTIwMDEwMTAwMDAwMFoY
DzIxMDAwMTAxMDAwMDAwWjApMScwJQYDVQQDDB5GcmFua2VuU2VhcmNoIE9SSUdJ
TiB0ZXN0IGxlYWYwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAARey+TRpjMKRMj3
75UdS/Fl5sa3Ie+tqYX7QWYbxuf9bIc0ZAxJmP9+N0sGzhpkouzYKrA2OE+4PZp5
sSeifVAyo4GSMIGPMAwGA1UdEwEB/wQCMAAwDgYDVR0PAQH/BAQDAgeAMBMGA1Ud
JQQMMAoGCCsGAQUFBwMBMDkGA1UdEQQyMDCCEm1vZGVsLWhvc3QuaW52YWxpZIIa
ZW1iZWRkaW5nLXByb3ZpZGVyLmludmFsaWQwHwYDVR0jBBgwFoAUtGejmZ6176Zx
3gAy8Iqcnl79oOEwCgYIKoZIzj0EAwIDRwAwRAIgVoY8PaFb2SMYCFEbr+3tLzdu
mSIaEmlvUQqi75kmxVkCIHdOwAioXzuMoGv2VmlqtZ24XWe23Ku2JjljTGoyXgN7
-----END CERTIFICATE-----
"""
TEST_KEY = """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAOhRANCAARey+TRpjMKRMj375UdS/Fl5sa3Ie+t
qYX7QWYbxuf9bIc0ZAxJmP9+N0sGzhpkouzYKrA2OE+4PZp5sSeifVAy
-----END PRIVATE KEY-----
"""


class ProbeError(Exception):
    """Only bounded, non-secret diagnostics belong in these errors."""

    def __init__(self, code, detail):
        super().__init__(f"proxy.binary.{code}: {detail}")
        self.code = code


def require(condition, code, detail):
    if not condition:
        raise ProbeError(code, detail)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "binary_type", "binary must remain a regular file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class Binary:
    def __init__(self, path, expected):
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                "binary_hash", "expected SHA-256 must be 64 lowercase hexadecimal digits")
        self.path = path.resolve(strict=True)
        self.initial_identity = identity(self.path)
        require(os.access(self.path, os.X_OK), "binary_type", "binary is not executable")
        self.expected = expected
        self.check(deep=True)

    def check(self, deep=False):
        require(identity(self.path) == self.initial_identity,
                "binary_changed", "supplied executable changed during qualification")
        if deep:
            require(digest_file(self.path) == self.expected,
                    "binary_hash", "supplied executable does not match the expected SHA-256")
            require(identity(self.path) == self.initial_identity,
                    "binary_changed", "supplied executable changed during hashing")


def write_private(path, data):
    with path.open("xb") as output:
        os.chmod(path, 0o600)
        output.write(data)


def child_environment(directory):
    """No inherited config, proxy, loader-injection, TLS-trust, or keylog knobs."""
    directories = {name: directory / name for name in ("home", "config", "cache", "data", "tmp", "models")}
    for path in directories.values():
        path.mkdir(mode=0o700)
    write_private(directory / "config.toml", b"# Private default configuration for executable qualification.\n")
    return {
        "PATH": os.defpath, "HOME": str(directories["home"]),
        "XDG_CONFIG_HOME": str(directories["config"]),
        "XDG_CACHE_HOME": str(directories["cache"]),
        "XDG_DATA_HOME": str(directories["data"]), "TMPDIR": str(directories["tmp"]),
        "FRANKENSEARCH_CHECK_UPDATES": "0", "FSFS_CHECK_UPDATES": "0",
        "NO_COLOR": "1", "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
    }


def stop_process(process):
    # Every invocation starts its own session. Signal only that owned process
    # group, then reap the direct child; this is not a general descendant audit.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def bounded_process(command, env, cwd, deadline):
    started = time.monotonic()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    outcome = {"stopped_by_helper": None}
    process = subprocess.Popen(command, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    with selectors.DefaultSelector() as selector:
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            selector.register(stream, selectors.EVENT_READ, name)
        try:
            while selector.get_map() or process.poll() is None:
                remaining = deadline - (time.monotonic() - started)
                if remaining <= 0:
                    outcome["stopped_by_helper"] = "deadline"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fd, 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    buffers[key.data].extend(chunk)
                    if (sum(map(len, buffers.values())) > MAX_OUTPUT_BYTES
                            or sum(value.count(b"\n") for value in buffers.values()) > MAX_OUTPUT_LINES):
                        outcome["stopped_by_helper"] = "output_limit"
                        break
                if outcome["stopped_by_helper"]:
                    break
        finally:
            if outcome["stopped_by_helper"] or process.poll() is None:
                stop_process(process)
            process.stdout.close()
            process.stderr.close()
    outcome.update(returncode=process.wait(timeout=2), elapsed_seconds=round(time.monotonic() - started, 3))
    return outcome, {name: bytes(value) for name, value in buffers.items()}


def envelope(raw, success):
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError):
        raise ProbeError("cli_output", "expected one structured fsfs JSON envelope") from None
    require(isinstance(value, dict) and value.get("ok") is success,
            "cli_output", "unexpected fsfs success/error envelope")
    return value


def validate_model_payload(raw, operation, state):
    value = envelope(raw, True).get("data", {})
    require(isinstance(value, dict) and value.get("operation") == operation,
            "model_command", "unexpected model operation")
    models = value.get("models", [])
    require(isinstance(models, list) and len(models) == 1 and isinstance(models[0], dict)
            and models[0].get("id") == MODEL and models[0].get("state") == state,
            "model_command", "expected exactly the selected registered model and state")
    if state != "missing":
        require(models[0].get("verified") is True, "model_command", "model was not reported verified")
    return models[0]


def validate_application_refusal(result, raw):
    require(result["stopped_by_helper"] is None and result["returncode"] > 0,
            "application_refusal", "refusal must be a normal nonzero fsfs exit, not a timeout or signal")
    error = envelope(raw, False).get("error", {})
    require(isinstance(error, dict) and error.get("code") == "model_load_failed"
            and error.get("exit_code") == result["returncode"],
            "application_refusal", "expected a typed model transport failure, not a CLI/config failure")
    return error["code"]


def receive_exact(stream, count, deadline=None):
    data = bytearray()
    while len(data) < count:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            require(remaining > 0, "peer_deadline", "proxy handshake exceeded its whole-exchange deadline")
            stream.settimeout(min(3, remaining))
        chunk = stream.recv(count - len(data))
        require(chunk, "peer_protocol", "client ended the proxy exchange prematurely")
        data.extend(chunk)
    return bytes(data)


def connect_origin(host, port):
    # Only --provision calls this. The client still validates origin TLS; this
    # proxy neither decrypts nor substitutes any upstream response bytes.
    require(port == 443 and (host == ORIGIN or host.endswith(".huggingface.co") or host.endswith(".hf.co")),
            "upstream_destination", "live proxy destination is outside the pinned model's HTTPS hosts")
    return socket.create_connection((host, port), timeout=10)


class ProxyPeer:
    """A small authenticated loopback proxy peer; never a replacement client."""

    def __init__(self, scheme, action, credentials, tls_files=None, deadline=10):
        self.scheme, self.action = scheme, action
        self.username, self.password = credentials
        self.tls_files, self.deadline = tls_files, deadline
        self.observations, self.errors = [], []
        self.stop = threading.Event()
        self.active_lock = threading.Lock()
        self.active_streams = []
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.listener.settimeout(0.1)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"{self.scheme}://{self.username}:{self.password}@127.0.0.1:{self.port}"

    @property
    def canaries(self):
        credentials = f"{self.username}:{self.password}".encode()
        return (self.username.encode(), self.password.encode(), credentials, base64.b64encode(credentials))

    def close(self):
        self.stop.set()
        self.listener.close()
        # Give an already-read terminal alert/refusal a short chance to finish,
        # then break an active recv/handshake/relay rather than only closing accept.
        self.thread.join(timeout=0.2)
        if self.thread.is_alive():
            with self.active_lock:
                active = list(self.active_streams)
            for stream in active:
                try:
                    stream.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                stream.close()
            self.thread.join(timeout=1)
        require(not self.thread.is_alive(), "peer_cleanup", "proxy peer did not terminate within its bound")

    def track(self, stream):
        with self.active_lock:
            self.active_streams.append(stream)
        return stream

    def receive(self, stream, count):
        return receive_exact(stream, count, self.handshake_deadline)

    def serve(self):
        while not self.stop.is_set():
            try:
                stream, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self.stop.is_set():
                    self.errors.append("listener_io")
                return
            if len(self.observations) >= MAX_PROXY_CONNECTIONS:
                stream.close()
                self.errors.append("connection_limit")
                return
            observed = {"protocol": self.scheme, "authenticated": False}
            self.observations.append(observed)
            self.handshake_deadline = time.monotonic() + min(5, self.deadline)
            self.track(stream)
            with stream:
                stream.settimeout(3)
                try:
                    if self.action == "trap":
                        continue
                    if self.action == "proxy-tls":
                        self.untrusted_tls(stream, observed)
                        continue
                    if self.scheme == "http":
                        host, port = self.connect_handshake(stream, observed)
                    else:
                        destination = self.socks_handshake(stream, observed)
                        if destination is None:
                            continue
                        host, port = destination
                    if self.action == "refuse":
                        if self.scheme == "http":
                            stream.sendall(b"HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                        else:
                            stream.sendall(b"\x05\x05\x00\x01\x7f\x00\x00\x01\x00\x00")
                        observed["refusal_sent"] = True
                        continue
                    if self.action == "forward":
                        with connect_origin(host, port) as upstream:
                            self.track(upstream)
                            self.tunnel_ready(stream)
                            self.forward(stream, upstream, observed)
                    else:
                        require(self.action == "tls", "peer_action", "unknown peer action")
                        self.tunnel_ready(stream)
                        self.untrusted_tls(stream, observed)
                except ProbeError as error:
                    self.errors.append(error.code)
                except (OSError, ValueError):
                    if not self.stop.is_set():
                        self.errors.append("peer_io")
                finally:
                    with self.active_lock:
                        self.active_streams.clear()

    def connect_handshake(self, stream, observed):
        data = bytearray()
        while not data.endswith(b"\r\n\r\n"):
            require(len(data) < MAX_HEADER_BYTES, "peer_protocol", "CONNECT headers exceeded bound")
            data.extend(self.receive(stream, 1))
        lines = bytes(data[:-4]).split(b"\r\n")
        parts = lines[0].split(b" ")
        require(len(parts) == 3 and parts[0] == b"CONNECT" and parts[2] == b"HTTP/1.1",
                "peer_protocol", "expected an HTTP CONNECT request")
        authority = parts[1].decode("ascii")
        match = re.fullmatch(r"([a-zA-Z0-9.-]+):([0-9]+)", authority)
        require(match is not None, "peer_protocol", "invalid CONNECT authority")
        headers = [line.split(b":", 1) for line in lines[1:]]
        require(all(len(header) == 2 for header in headers), "peer_protocol", "malformed CONNECT header")
        auth = [value.strip() for key, value in headers if key.lower() == b"proxy-authorization"]
        expected = b"Basic " + base64.b64encode(f"{self.username}:{self.password}".encode())
        require(auth == [expected], "peer_credentials", "CONNECT credentials were missing or incorrect")
        observed.update(authenticated=True, host=match[1].lower(), port=int(match[2]))
        return observed["host"], observed["port"]

    def socks_handshake(self, stream, observed):
        version, count = self.receive(stream, 2)
        require(version == 5 and count > 0, "peer_protocol", "expected a SOCKS5 greeting")
        require(2 in self.receive(stream, count), "peer_protocol", "SOCKS username/password authentication was not offered")
        stream.sendall(b"\x05\x02")
        version, count = self.receive(stream, 2)
        require(version == 1, "peer_protocol", "expected SOCKS username/password authentication")
        username = self.receive(stream, count)
        password = self.receive(stream, self.receive(stream, 1)[0])
        require(username == self.username.encode() and password == self.password.encode(),
                "peer_credentials", "SOCKS credentials were missing or incorrect")
        observed["authenticated"] = True
        if self.action == "auth-refuse":
            stream.sendall(b"\x01\x01")
            observed["authentication_refusal_sent"] = True
            return None
        stream.sendall(b"\x01\x00")
        require(self.receive(stream, 4) == b"\x05\x01\x00\x03",
                "peer_protocol", "SOCKS CONNECT must transmit a domain name (ATYP=3)")
        host = self.receive(stream, self.receive(stream, 1)[0]).decode("ascii")
        require(re.fullmatch(r"[a-zA-Z0-9.-]+", host), "peer_protocol", "invalid SOCKS hostname")
        port = int.from_bytes(self.receive(stream, 2), "big")
        observed.update(host=host.lower(), port=port, address_type=3)
        return observed["host"], port

    def tunnel_ready(self, stream):
        if self.scheme == "http":
            stream.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        else:
            stream.sendall(b"\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x00")

    def untrusted_tls(self, stream, observed):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(*self.tls_files)
        context.set_servername_callback(lambda _stream, name, _context: observed.update(tls_sni=name))
        observed["tls_handshake_completed"] = False
        try:
            with context.wrap_socket(stream, server_side=True, do_handshake_on_connect=False) as secured:
                self.track(secured)
                secured.settimeout(max(0.01, min(3, self.handshake_deadline - time.monotonic())))
                secured.do_handshake()
                observed["tls_handshake_completed"] = True
                # Successful acceptance itself violates this probe. Do not serve
                # any model bytes, even to a client that ignores TLS validation.
                observed["application_request_bytes"] = len(secured.recv(4096))
        except ssl.SSLError as error:
            observed["tls_alert"] = error.reason if error.reason in TLS_REFUSAL_ALERTS else "other_tls_error"

    def forward(self, downstream, upstream, observed):
        started = time.monotonic()
        observed.update(relayed_from_origin=0, relayed_to_origin=0)
        upstream.settimeout(3)
        with selectors.DefaultSelector() as selector:
            selector.register(downstream, selectors.EVENT_READ, (upstream, "relayed_to_origin"))
            selector.register(upstream, selectors.EVENT_READ, (downstream, "relayed_from_origin"))
            while selector.get_map() and not self.stop.is_set():
                require(time.monotonic() - started < self.deadline,
                        "peer_deadline", "live tunnel exceeded its deadline")
                for key, _ in selector.select(0.1):
                    target, counter = key.data
                    try:
                        data = key.fileobj.recv(32768)
                    except ConnectionResetError:
                        observed["relay_closed"] = "read_reset"
                        return
                    if not data:
                        selector.unregister(key.fileobj)
                        try:
                            target.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        continue
                    require(observed[counter] + len(data) <= MAX_TUNNEL_BYTES,
                            "peer_byte_limit", "live tunnel exceeded the pinned model transfer bound")
                    try:
                        target.sendall(data)
                    except (BrokenPipeError, ConnectionResetError):
                        observed["relay_closed"] = "write_closed"
                        return
                    # A reset ends this tunnel, never proves acquisition. Count
                    # only fully sent chunks; CLI success and pinned files must
                    # independently pass before the live case can succeed.
                    observed[counter] += len(data)


def validate_peer(peer, action):
    require(not peer.errors, "peer_exchange", "proxy peer observed an invalid or incomplete exchange")
    # The production downloader retries ModelLoadFailed up to three times;
    # it does not expose a CLI retry override. Validate every actual exchange.
    require(1 <= len(peer.observations) <= 4, "proxy_contact", "expected one to four requests to the selected proxy")
    for observed in peer.observations:
        if action == "proxy-tls":
            require(not observed.get("authenticated") and not observed.get("tls_handshake_completed")
                    and observed.get("tls_alert") in TLS_REFUSAL_ALERTS,
                    "proxy_tls_refusal", "expected outer proxy certificate refusal before HTTP proxy authentication")
            continue
        require(observed.get("authenticated"), "proxy_auth", "proxy did not receive the expected credentials")
        if action == "auth-refuse":
            require(observed.get("authentication_refusal_sent"), "proxy_auth", "SOCKS authentication was not refused")
            continue
        require(observed.get("host") == ORIGIN and observed.get("port") == 443,
                "proxy_destination", "proxy did not receive the pinned HTTPS model authority")
        if peer.scheme != "http":
            require(observed.get("address_type") == 3, "proxy_dns", "SOCKS did not transmit the destination hostname")
        if action == "tls":
            require(observed.get("tls_sni") == ORIGIN and not observed.get("tls_handshake_completed")
                    and observed.get("tls_alert") in TLS_REFUSAL_ALERTS,
                    "tls_refusal", "expected a client certificate-refusal alert with the origin SNI")
        else:
            require(observed.get("refusal_sent"), "proxy_refusal", "proxy did not send its deliberate refusal")


def no_model_bytes(directory):
    # A failed acquisition may leave an empty unique staging directory. No
    # installed model, partial file, symlink, or verified receipt is acceptable.
    require(directory.is_dir() and not directory.is_symlink(), "model_publication",
            "refused transport replaced a model/staging directory with a file or link")
    for path in directory.rglob("*"):
        require(path.is_dir() and not path.is_symlink(), "model_publication",
                "refused transport left a file, symlink, or model receipt in the private model area")


def verify_model_files(directory, files=None):
    expected = MODEL_FILES if files is None else files
    result = {}
    for path in (directory.parent, directory):
        require(path.exists() and stat.S_ISDIR(path.lstat().st_mode),
                "model_integrity", "model and output directories must remain directories without symlinks")
    for name, (size, digest) in expected.items():
        path = directory / name
        require(path.exists() and stat.S_ISREG(path.lstat().st_mode)
                and path.stat().st_size == size and digest_file(path) == digest,
                "model_integrity", "registered model bytes failed pinned size/SHA-256 verification")
        result[name] = {"bytes": size, "sha256": digest}
    return result


class Qualification:
    def __init__(self, binary, work, deadline, provision_deadline):
        self.binary, self.work = binary, work
        self.deadline, self.provision_deadline = deadline, provision_deadline
        self.results = []
        self.canaries = []
        self.tls_files = (str(work / "public-test-origin.pem"), str(work / "public-test-origin-key.pem"))
        write_private(Path(self.tls_files[0]), TEST_CERTIFICATE.encode())
        write_private(Path(self.tls_files[1]), TEST_KEY.encode())

    def environment(self, name):
        directory = self.work / name
        directory.mkdir(mode=0o700)
        env = child_environment(directory)
        return directory, env

    def invoke(self, name, directory, env, args, canaries=(), deadline=None):
        self.binary.check()
        result, raw = bounded_process([str(self.binary.path), *args], env, directory,
                                      deadline or self.deadline)
        self.binary.check()
        leaked = any(canary in output for canary in canaries for output in raw.values())
        for channel, output in raw.items():
            for canary in sorted(canaries, key=len, reverse=True):
                output = output.replace(canary, b"[redacted-proxy-canary]")
            write_private(directory / f"{name}.{channel}.txt", output)
        result.update(command=args, output_canary_leaked=leaked)
        return result, raw

    def arguments(self, directory, *extra):
        return ["download-models", "--model", MODEL, "--output", str(directory / "models"),
                "--config", str(directory / "config.toml"), "--format", "json", "--quiet", *extra]

    def preflight(self, expected_version):
        directory, env = self.environment("preflight")
        result, raw = self.invoke("version", directory, env, ["version"])
        self.results.append({"case": "version", **result})
        require(result["stopped_by_helper"] is None and result["returncode"] == 0,
                "version", "supplied fsfs failed its bounded version command")
        match = re.fullmatch(rb"fsfs ([0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?) \(frankensearch [^\r\n]+\)\s*", raw["stdout"])
        require(match is not None, "version", "supplied executable did not identify itself as fsfs")
        reported_version = match[1].decode()
        require(expected_version is None or reported_version == expected_version.removeprefix("v"),
                "version", "reported fsfs version differs from the requested version")
        result, raw = self.invoke("list", directory, env, self.arguments(directory, "--list", "--offline"))
        self.results.append({"case": "registered-model-cold-cache", **result})
        require(result["stopped_by_helper"] is None and result["returncode"] == 0,
                "model_preflight", "fsfs could not list the registered model in its private empty cache")
        validate_model_payload(raw["stdout"], "list", "missing")
        return reported_version

    def run_case(self, name, scheme, action, configure):
        directory, env = self.environment(name)
        credentials = ("proxy-user-" + secrets.token_hex(8), "proxy-secret-" + secrets.token_hex(12))
        primary = ProxyPeer(scheme, action, credentials, self.tls_files, self.deadline)
        alternate = ProxyPeer("http", "trap", credentials, deadline=self.deadline)
        self.canaries.extend(primary.canaries)
        env.update(configure(primary.url, alternate.url))
        record = {"case": name, "protocol": scheme, "action": action}
        self.results.append(record)
        try:
            result, raw = self.invoke("download", directory, env, self.arguments(directory, "--online"), primary.canaries)
            record.update(result)
        finally:
            try:
                primary.close()
            finally:
                try:
                    alternate.close()
                finally:
                    record.update(proxy_observations=primary.observations, proxy_errors=primary.errors,
                                  alternate_proxy_contacts=len(alternate.observations))
        require(not result["output_canary_leaked"], "credential_leak", "fsfs output exposed a proxy credential canary")
        # Stages are siblings of the requested output directory, so inspect the
        # whole private case area excluding the known environment/output files.
        model_area = [path for path in directory.iterdir()
                      if path.name == "models" or path.name.startswith(".models-download-")]
        for path in model_area:
            no_model_bytes(path)
        require(not (directory / "models" / MODEL_DIRECTORY).exists(),
                "model_publication", "refused transport installed the model")
        validate_peer(primary, action)
        require(not alternate.observations and not alternate.errors,
                "proxy_precedence", "lower-priority proxy was contacted")
        record["application_error"] = validate_application_refusal(result, raw["stdout"])
        record["result"] = "PASS"

    def wire_cases(self):
        cases = (
            ("https-uppercase-connect", "http", "refuse", lambda p, a: {"HTTPS_PROXY": p, "ALL_PROXY": a}),
            ("https-lowercase-precedence", "http", "refuse", lambda p, a: {"https_proxy": p, "HTTPS_PROXY": a, "ALL_PROXY": a}),
            ("all-lowercase-socks5", "socks5", "refuse", lambda p, a: {"all_proxy": p, "ALL_PROXY": a}),
            ("all-uppercase-socks5h-no-proxy-nonmatch", "socks5h", "refuse", lambda p, a: {"ALL_PROXY": p, "NO_PROXY": "unrelated.invalid,huggingface.co:444"}),
            ("socks-authentication-refusal", "socks5h", "auth-refuse", lambda p, a: {"all_proxy": p, "ALL_PROXY": a}),
            ("connect-untrusted-tls", "http", "tls", lambda p, a: {"HTTPS_PROXY": p, "ALL_PROXY": a}),
            ("socks5-untrusted-tls", "socks5", "tls", lambda p, a: {"all_proxy": p, "ALL_PROXY": a}),
            ("socks5h-untrusted-tls", "socks5h", "tls", lambda p, a: {"all_proxy": p, "ALL_PROXY": a}),
            ("https-proxy-untrusted-tls", "https", "proxy-tls", lambda p, a: {"HTTPS_PROXY": p, "ALL_PROXY": a}),
        )
        for case in cases:
            self.run_case(*case)

    def provision(self):
        for scheme, bypass in (("http", False), ("socks5", False), ("socks5h", False), ("http", True)):
            name = "live-no-proxy-match" if bypass else f"live-{scheme}"
            directory, env = self.environment(name)
            credentials = ("proxy-user-" + secrets.token_hex(8), "proxy-secret-" + secrets.token_hex(12))
            peer = ProxyPeer(scheme, "trap" if bypass else "forward", credentials, deadline=self.provision_deadline)
            self.canaries.extend(peer.canaries)
            env["HTTPS_PROXY" if scheme == "http" else "ALL_PROXY"] = peer.url
            if bypass:
                env.update(ALL_PROXY=peer.url, NO_PROXY="huggingface.co,.hf.co")
            record = {"case": name, "protocol": scheme, "action": "verified-provisioning",
                      "route_evidence": "matching NO_PROXY with no configured proxy contact" if bypass
                      else "authenticated proxy exchanges and encrypted traffic; final bytes verified independently"}
            self.results.append(record)
            try:
                result, raw = self.invoke("download", directory, env, self.arguments(directory, "--online"), peer.canaries, self.provision_deadline)
                record["download"] = result
                require(result["stopped_by_helper"] is None and result["returncode"] == 0 and not result["output_canary_leaked"],
                        "provisioning", "live model acquisition did not complete cleanly")
                validate_model_payload(raw["stdout"], "download", "downloaded")
                record["files"] = verify_model_files(directory / "models" / MODEL_DIRECTORY)
                if bypass:
                    require(not peer.observations, "no_proxy_bypass", "matching NO_PROXY acquisition contacted the configured proxy")
                else:
                    require(peer.observations and all(item.get("authenticated") for item in peer.observations)
                            and sum(item.get("relayed_from_origin", 0) for item in peer.observations) >= sum(item[0] for item in MODEL_FILES.values()),
                            "provisioning_route", "proxy did not record sufficient authenticated origin traffic")
                contact_count = len(peer.observations)
                for label, flags, operation, state in (
                    ("verify", ("--verify", "--offline"), "verify", "verified"),
                    ("warm-offline", ("--offline",), "download", "cached"),
                ):
                    result, raw = self.invoke(label, directory, env, self.arguments(directory, *flags), peer.canaries)
                    record[label] = result
                    require(result["stopped_by_helper"] is None and result["returncode"] == 0 and not result["output_canary_leaked"],
                            "offline_replay", "verification or warm offline replay did not complete cleanly")
                    validate_model_payload(raw["stdout"], operation, state)
                    require(len(peer.observations) == contact_count,
                            "offline_proxy_contact", "verification or warm offline replay contacted the proxy")
                verify_model_files(directory / "models" / MODEL_DIRECTORY)
            finally:
                try:
                    peer.close()
                finally:
                    record.update(proxy_observations=peer.observations, proxy_errors=peer.errors)
            require(not peer.errors, "provisioning_peer", "live proxy observed a failed exchange")
            require(len(peer.observations) == contact_count,
                    "offline_proxy_contact", "a late verification/offline proxy contact was observed")
            record["result"] = "PASS"


def finite_deadline(value):
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("deadline must be a finite number") from None
    if not 0.25 <= seconds <= 600:
        raise argparse.ArgumentTypeError("deadline must be between 0.25 and 600 seconds")
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-version")
    parser.add_argument("--work-dir", type=Path, required=True, help="new retained private directory; must not already exist")
    parser.add_argument("--deadline-seconds", type=finite_deadline, default=20)
    parser.add_argument("--provision", action="store_true", help="explicitly enable live upstream downloads through three proxy transports and matching NO_PROXY (four 31 MB acquisitions)")
    parser.add_argument("--provision-deadline-seconds", type=finite_deadline, default=180)
    args = parser.parse_args(argv)
    receipt = {
        "result": "FAIL", "claim": "proxy-wire-security-only", "host": {"platform": sys.platform, "machine": platform.machine()},
        "expected_binary_sha256": args.expected_sha256, "model": MODEL, "model_revision": MODEL_REVISION,
        "provisioning_requested": args.provision, "cases": [],
        "unqualified": ["direct-egress isolation or observation", "absence of direct fallback connections",
                        "direct-route observation for matching NO_PROXY", "independent trusted-chain hostname validation",
                        "cryptographic attribution of final model bytes to proxy observations",
                        "native-versus-emulated execution on this host; execution on other platforms",
                        "installer provisioning of the default model pair",
                        "signed-release publication and source provenance"],
    }
    work = args.work_dir.resolve()
    work_created = False
    binary = None
    qualification = None
    try:
        require(os.name == "posix", "platform", "this bounded process/peer runner requires a POSIX host")
        binary = Binary(args.binary, args.expected_sha256)
        require(not work.exists(), "work_directory", "work directory already exists; prior evidence is never overwritten")
        work.mkdir(mode=0o700)
        work_created = True
        receipt.update(binary=str(binary.path), work_directory=str(work))
        qualification = Qualification(binary, work, args.deadline_seconds, args.provision_deadline_seconds)
        receipt["cases"] = qualification.results
        receipt["reported_version"] = qualification.preflight(args.expected_version)
        qualification.wire_cases()
        if args.provision:
            qualification.provision()
            receipt["claim"] = "proxy-wire-security-and-live-pinned-model-provisioning"
        else:
            receipt["unqualified"].append("complete model provisioning (requires explicit --provision)")
            receipt["unqualified"].append("successful matching NO_PROXY acquisition (requires explicit --provision)")
        receipt.update(result="PASS", observed_binary_sha256=binary.expected)
    except (ProbeError, OSError, subprocess.SubprocessError) as error:
        # Never echo an opaque OS/server/child exception that could contain
        # proxy credentials. Structured ProbeError details are authored above.
        receipt["error"] = str(error) if isinstance(error, ProbeError) else f"proxy.binary.execution: {type(error).__name__}"
    finally:
        if binary is not None:
            try:
                binary.check(deep=True)
                receipt["observed_binary_sha256"] = binary.expected
            except (ProbeError, OSError):
                receipt.update(result="FAIL", error="proxy.binary.binary_changed: executable identity/hash check failed")
        encoded = (json.dumps(receipt, indent=2) + "\n").encode()
        if qualification is not None:
            for canary in sorted(qualification.canaries, key=len, reverse=True):
                encoded = encoded.replace(canary, b"[redacted-proxy-canary]")
        if work_created:
            try:
                write_private(work / "receipt.json", encoded)
            except OSError:
                # Preserve the already-redacted representation on this error
                # path too, instead of reintroducing raw observation strings.
                receipt = json.loads(encoded)
                receipt.update(result="FAIL", error="proxy.binary.receipt_write: could not preserve the result")
                encoded = (json.dumps(receipt, indent=2) + "\n").encode()
        print(encoded.decode(), end="")
    return 0 if receipt["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
