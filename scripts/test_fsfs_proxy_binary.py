#!/usr/bin/env python3
"""Local peer/supervision regressions, never fsfs executable qualification.

These tests use tiny protocol clients and owned Python children. They do not
build or retrieve fsfs, resolve/connect to upstream hosts, or validate a release.
Fixture directories are intentionally retained. Set TMPDIR to a writable volume.
"""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import ssl
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import check_fsfs_proxy_binary as check


CREDENTIALS = ("fixture-user", "fixture-secret")


def work_directory():
    return Path(tempfile.mkdtemp(prefix="fsfs-proxy-peer-test-"))


def connect_client(peer):
    stream = socket.create_connection(("127.0.0.1", peer.port), timeout=3)
    auth = base64.b64encode(":".join(CREDENTIALS).encode())
    stream.sendall(b"CONNECT huggingface.co:443 HTTP/1.1\r\nHost: huggingface.co:443\r\n"
                   b"Proxy-Authorization: Basic " + auth + b"\r\n\r\n")
    response = bytearray()
    while not response.endswith(b"\r\n\r\n"):
        response.extend(check.receive_exact(stream, 1))
    return stream, bytes(response)


def socks_client(peer, reject_auth=False, address_type=3):
    stream = socket.create_connection(("127.0.0.1", peer.port), timeout=3)
    try:
        stream.sendall(b"\x05\x01\x02")
        if check.receive_exact(stream, 2) != b"\x05\x02":
            raise AssertionError("fixture peer did not select username/password authentication")
        username, password = (item.encode() for item in CREDENTIALS)
        stream.sendall(b"\x01" + bytes([len(username)]) + username + bytes([len(password)]) + password)
        reply = check.receive_exact(stream, 2)
        if reject_auth:
            return stream, reply
        if reply != b"\x01\x00":
            raise AssertionError("fixture peer unexpectedly rejected authentication")
        host = b"huggingface.co"
        stream.sendall(bytes([5, 1, 0, address_type]) + bytes([len(host)]) + host + b"\x01\xbb")
        return stream, check.receive_exact(stream, 10)
    except BaseException:
        stream.close()
        raise


class PeerTests(unittest.TestCase):
    def test_authenticated_connect_refusal_and_default_retries(self):
        peer = check.ProxyPeer("http", "refuse", CREDENTIALS)
        try:
            for _ in range(4):
                stream, response = connect_client(peer)
                stream.close()
                self.assertTrue(response.startswith(b"HTTP/1.1 407"))
        finally:
            peer.close()
        check.validate_peer(peer, "refuse")
        self.assertEqual(len(peer.observations), 4)
        self.assertNotIn("fixture-secret", json.dumps(peer.observations))
        peer.observations.append(peer.observations[0].copy())
        with self.assertRaisesRegex(check.ProbeError, "proxy_contact"):
            check.validate_peer(peer, "refuse")

    def test_connect_missing_credentials_is_not_a_valid_probe(self):
        peer = check.ProxyPeer("http", "refuse", CREDENTIALS)
        try:
            with socket.create_connection(("127.0.0.1", peer.port), timeout=3) as stream:
                stream.sendall(b"CONNECT huggingface.co:443 HTTP/1.1\r\n\r\n")
                self.assertEqual(stream.recv(1), b"")
        finally:
            peer.close()
        with self.assertRaisesRegex(check.ProbeError, "peer_exchange"):
            check.validate_peer(peer, "refuse")

    def test_socks_spellings_require_remote_hostname_and_authentication(self):
        for scheme in ("socks5", "socks5h"):
            with self.subTest(scheme=scheme):
                peer = check.ProxyPeer(scheme, "refuse", CREDENTIALS)
                try:
                    stream, response = socks_client(peer)
                    stream.close()
                    self.assertEqual(response[:4], b"\x05\x05\x00\x01")
                finally:
                    peer.close()
                check.validate_peer(peer, "refuse")
                self.assertEqual(peer.observations[0]["address_type"], 3)
                self.assertNotIn("fixture-secret", json.dumps(peer.observations))

    def test_socks_ip_address_type_cannot_pass_remote_dns_probe(self):
        peer = check.ProxyPeer("socks5", "refuse", CREDENTIALS)
        try:
            with self.assertRaises((check.ProbeError, ConnectionResetError)):
                stream, _ = socks_client(peer, address_type=1)
                stream.close()
        finally:
            peer.close()
        with self.assertRaisesRegex(check.ProbeError, "peer_exchange"):
            check.validate_peer(peer, "refuse")

    def test_socks_authentication_refusal_is_distinct_from_destination_refusal(self):
        peer = check.ProxyPeer("socks5h", "auth-refuse", CREDENTIALS)
        try:
            stream, response = socks_client(peer, reject_auth=True)
            stream.close()
            self.assertEqual(response, b"\x01\x01")
        finally:
            peer.close()
        check.validate_peer(peer, "auth-refuse")
        self.assertNotIn("host", peer.observations[0])
        with self.assertRaisesRegex(check.ProbeError, "proxy_destination"):
            check.validate_peer(peer, "refuse")

    def tls_files(self):
        root = work_directory()
        certificate, key = root / "test.pem", root / "test.key"
        check.write_private(certificate, check.TEST_CERTIFICATE.encode())
        check.write_private(key, check.TEST_KEY.encode())
        return str(certificate), str(key)

    def test_real_tls_certificate_refusal_after_connect_and_socks(self):
        for scheme in ("http", "socks5", "socks5h"):
            with self.subTest(scheme=scheme):
                peer = check.ProxyPeer(scheme, "tls", CREDENTIALS, self.tls_files())
                stream = None
                try:
                    stream, _ = connect_client(peer) if scheme == "http" else socks_client(peer)
                    # Explicitly empty roots in this fixture client; no root is
                    # ever installed into fsfs by the production helper.
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                    with self.assertRaises(ssl.SSLCertVerificationError):
                        context.wrap_socket(stream, server_hostname=check.ORIGIN)
                finally:
                    if stream:
                        stream.close()
                    peer.close()
                check.validate_peer(peer, "tls")
                self.assertEqual(peer.observations[0]["tls_sni"], check.ORIGIN)

    def test_https_proxy_certificate_refusal_happens_before_proxy_credentials(self):
        peer = check.ProxyPeer("https", "proxy-tls", CREDENTIALS, self.tls_files())
        try:
            with socket.create_connection(("127.0.0.1", peer.port), timeout=3) as stream:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                with self.assertRaises(ssl.SSLCertVerificationError):
                    context.wrap_socket(stream, server_hostname="127.0.0.1")
        finally:
            peer.close()
        check.validate_peer(peer, "proxy-tls")
        self.assertFalse(peer.observations[0]["authenticated"])
        self.assertNotIn("host", peer.observations[0])

    def test_peer_close_interrupts_an_active_handshake(self):
        peer = check.ProxyPeer("socks5", "refuse", CREDENTIALS)
        with socket.create_connection(("127.0.0.1", peer.port), timeout=3) as stream:
            stream.sendall(b"\x05\x01\x02")
            self.assertEqual(check.receive_exact(stream, 2), b"\x05\x02")
            started = time.monotonic()
            peer.close()
            self.assertLess(time.monotonic() - started, 1)
        self.assertFalse(peer.thread.is_alive())

    def test_whole_handshake_deadline_bounds_slow_progress(self):
        reader, writer = socket.socketpair()
        stop = threading.Event()

        def drip():
            with writer:
                while not stop.wait(0.02):
                    try:
                        writer.sendall(b"x")
                    except OSError:
                        return

        thread = threading.Thread(target=drip, daemon=True)
        thread.start()
        try:
            with reader:
                started = time.monotonic()
                with self.assertRaises((check.ProbeError, TimeoutError)):
                    check.receive_exact(reader, 100, started + 0.1)
                self.assertLess(time.monotonic() - started, 0.5)
        finally:
            stop.set()
            thread.join(timeout=1)
        self.assertFalse(thread.is_alive())

    def test_accepting_untrusted_tls_is_a_probe_failure(self):
        peer = check.ProxyPeer("http", "tls", CREDENTIALS, self.tls_files())
        try:
            stream, _ = connect_client(peer)
            # Deliberately broken test client, not a trust option in the helper.
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            with context.wrap_socket(stream, server_hostname=check.ORIGIN) as secured:
                secured.sendall(b"GET / HTTP/1.1\r\n\r\n")
                self.assertEqual(secured.recv(1), b"")
        finally:
            peer.close()
        with self.assertRaisesRegex(check.ProbeError, "tls_refusal"):
            check.validate_peer(peer, "tls")

    def test_forwarding_is_opaque_and_uses_only_an_injected_loopback_upstream(self):
        for scheme in ("http", "socks5", "socks5h"):
            with self.subTest(scheme=scheme):
                listener = socket.socket()
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                listener.settimeout(3)
                observations = []

                def origin():
                    with listener:
                        stream, _ = listener.accept()
                        with stream:
                            stream.settimeout(3)
                            observations.append(check.receive_exact(stream, 5))
                            stream.sendall(b"world")

                thread = threading.Thread(target=origin, daemon=True)
                thread.start()
                address = listener.getsockname()
                with mock.patch.object(check, "connect_origin", side_effect=lambda _h, _p: socket.create_connection(address, timeout=3)) as connect:
                    peer = check.ProxyPeer(scheme, "forward", CREDENTIALS)
                    try:
                        stream, _ = connect_client(peer) if scheme == "http" else socks_client(peer)
                        with stream:
                            stream.sendall(b"hello")
                            self.assertEqual(check.receive_exact(stream, 5), b"world")
                            self.assertEqual(stream.recv(1), b"")
                    finally:
                        peer.close()
                    connect.assert_called_once_with(check.ORIGIN, 443)
                thread.join(timeout=4)
                self.assertFalse(thread.is_alive())
                self.assertEqual(observations, [b"hello"])
                self.assertEqual(peer.errors, [])
                self.assertEqual(peer.observations[0]["relayed_from_origin"], 5)

    def test_relay_reset_is_recorded_as_closure_without_claiming_transferred_bytes(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(3)
        address = listener.getsockname()
        release = threading.Event()

        def reset_origin():
            with listener:
                stream, _ = listener.accept()
                with stream:
                    release.wait(3)
                    stream.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))

        thread = threading.Thread(target=reset_origin, daemon=True)
        thread.start()
        with mock.patch.object(check, "connect_origin", side_effect=lambda _h, _p: socket.create_connection(address, timeout=3)):
            peer = check.ProxyPeer("http", "forward", CREDENTIALS)
            try:
                stream, _ = connect_client(peer)
                release.set()
                with stream:
                    self.assertEqual(stream.recv(1), b"")
            finally:
                release.set()
                peer.close()
        thread.join(timeout=4)
        self.assertFalse(thread.is_alive())
        self.assertEqual(peer.errors, [])
        self.assertEqual(peer.observations[0]["relay_closed"], "read_reset")
        self.assertEqual(peer.observations[0]["relayed_from_origin"], 0)


class ValidationTests(unittest.TestCase):
    def test_application_failure_requires_a_normal_typed_exit(self):
        raw = json.dumps({"ok": False, "error": {"code": "model_load_failed", "exit_code": 78}}).encode()
        good = {"returncode": 78, "stopped_by_helper": None}
        self.assertEqual(check.validate_application_refusal(good, raw), "model_load_failed")
        for bad in ({"returncode": -9, "stopped_by_helper": "deadline"},
                    {"returncode": 0, "stopped_by_helper": None},
                    {"returncode": 78, "stopped_by_helper": "output_limit"}):
            with self.assertRaises(check.ProbeError):
                check.validate_application_refusal(bad, raw)
        with self.assertRaisesRegex(check.ProbeError, "application_refusal"):
            check.validate_application_refusal(good, raw.replace(b"model_load_failed", b"invalid_config"))

    def test_model_preflight_rejects_warm_or_wrong_models(self):
        raw = {"ok": True, "data": {"operation": "list", "models": [{"id": check.MODEL, "state": "missing"}]}}
        check.validate_model_payload(json.dumps(raw).encode(), "list", "missing")
        raw["data"]["models"][0]["state"] = "cached"
        with self.assertRaises(check.ProbeError):
            check.validate_model_payload(json.dumps(raw).encode(), "list", "missing")

    def test_refusal_rejects_partial_bytes_receipts_and_symlinks(self):
        root = work_directory()
        check.no_model_bytes(root)
        (root / "empty-stage").mkdir()
        check.no_model_bytes(root)
        (root / "partial.tmp").write_bytes(b"partial")
        with self.assertRaisesRegex(check.ProbeError, "model_publication"):
            check.no_model_bytes(root)
        link = work_directory() / "models"
        link.symlink_to(work_directory(), target_is_directory=True)
        with self.assertRaisesRegex(check.ProbeError, "model_publication"):
            check.no_model_bytes(link)

    def test_model_integrity_verifies_bytes_and_refuses_same_size_corruption(self):
        root = work_directory()
        payload = b"fixture bytes"
        (root / "artifact").write_bytes(payload)
        manifest = {"artifact": (len(payload), hashlib.sha256(payload).hexdigest())}
        self.assertEqual(check.verify_model_files(root, manifest)["artifact"]["bytes"], len(payload))
        (root / "artifact").write_bytes(b"x" * len(payload))
        with self.assertRaisesRegex(check.ProbeError, "model_integrity"):
            check.verify_model_files(root, manifest)

    def test_live_model_verification_rejects_linked_model_and_output_directories(self):
        root = work_directory()
        output = root / "models"
        output.mkdir()
        model = output / "model"
        model.mkdir()
        payload = b"fixture bytes"
        (model / "artifact").write_bytes(payload)
        manifest = {"artifact": (len(payload), hashlib.sha256(payload).hexdigest())}
        model_link = output / "model-link"
        model_link.symlink_to(model, target_is_directory=True)
        output_link = root / "models-link"
        output_link.symlink_to(output, target_is_directory=True)
        for path in (model_link, output_link / "model"):
            with self.subTest(path=path), self.assertRaisesRegex(check.ProbeError, "model_integrity"):
                check.verify_model_files(path, manifest)

    def test_environment_excludes_ambient_proxy_configuration_and_trust(self):
        root = work_directory()
        ambient = {"HTTPS_PROXY": "http://secret", "SSL_CERT_FILE": "/secret", "SSLKEYLOGFILE": "/secret",
                   "FRANKENSEARCH_OFFLINE": "1", "LD_PRELOAD": "/secret", "FSFS_CHECK_UPDATES": "1"}
        with mock.patch.dict(os.environ, ambient):
            env = check.child_environment(root)
        for name in ambient:
            if name != "FSFS_CHECK_UPDATES":
                self.assertNotIn(name, env)
        self.assertEqual(env["FSFS_CHECK_UPDATES"], "0")
        self.assertEqual(env["HOME"], str(root / "home"))
        self.assertTrue((root / "config.toml").is_file())

    def test_binary_sha_mismatch_does_not_execute_or_create_work_directory(self):
        work = work_directory() / "run"
        with mock.patch.object(check, "bounded_process") as execute, mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            result = check.main(["--binary", sys.executable, "--expected-sha256", "0" * 64, "--work-dir", str(work)])
        self.assertEqual(result, 1)
        self.assertFalse(work.exists())
        execute.assert_not_called()
        self.assertIn("binary_hash", json.loads(output.getvalue())["error"])

    def test_binary_identity_change_is_rejected(self):
        root = work_directory()
        path = root / "fixture-executable"
        path.write_bytes(b"not executed")
        path.chmod(0o700)
        binary = check.Binary(path, check.digest_file(path))
        path.write_bytes(b"changed fixture")
        with self.assertRaisesRegex(check.ProbeError, "binary_changed"):
            binary.check()

    def test_deadline_rejects_nonfinite_or_unbounded_values(self):
        for value in ("nan", "inf", "-1", "0", "601"):
            with self.assertRaises(argparse.ArgumentTypeError):
                check.finite_deadline(value)
        self.assertEqual(check.finite_deadline("0.25"), 0.25)

    def test_process_deadline_and_output_cap_are_failures_not_refusals(self):
        root = work_directory()
        env = check.child_environment(root)
        result, _ = check.bounded_process([sys.executable, "-B", "-c", "import time; time.sleep(20)"], env, root, 0.25)
        self.assertEqual(result["stopped_by_helper"], "deadline")
        self.assertLess(result["elapsed_seconds"], 3)
        result, raw = check.bounded_process([sys.executable, "-B", "-c", "import os; os.write(1, b'x' * 300000)"], env, root, 3)
        self.assertEqual(result["stopped_by_helper"], "output_limit")
        self.assertLessEqual(len(raw["stdout"]), check.MAX_OUTPUT_BYTES + 4096)

    def test_canary_outputs_are_flagged_and_redacted_in_retained_logs(self):
        root = work_directory()
        binary = check.Binary(Path(sys.executable), check.digest_file(Path(sys.executable)))
        qualification = check.Qualification(binary, root, 3, 10)
        directory, env = qualification.environment("canary-test")
        secret = b"fixture-secret"
        result, _ = qualification.invoke("fixture", directory, env, ["-B", "-c", "print('fixture-secret')"], [secret])
        self.assertTrue(result["output_canary_leaked"])
        self.assertNotIn(secret, (directory / "fixture.stdout.txt").read_bytes())
        self.assertIn(b"[redacted-proxy-canary]", (directory / "fixture.stdout.txt").read_bytes())


if __name__ == "__main__":
    unittest.main()
