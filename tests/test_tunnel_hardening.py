"""Stage D: unit tests for tunnel + supply-chain hardening.

Runnable without Kaggle/TPU. `launch.py` imports cleanly (main guard);
kernel helpers are extracted from the shipped `serve_qwen38.py` source
via ast so the tests track the real code, not a copy (same pattern as
`tests/test_weights_source.py`).
"""
import ast
import contextlib
import io
import json
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import launch  # noqa: E402  (safe: main() is behind __main__ guard)

KERNEL = REPO / "qwen38-27b" / "kernel" / "serve_qwen38.py"


def load_kernel_funcs(*names):
    """Exec the named top-level functions from serve_qwen38.py in a sandbox."""
    import hashlib as _hashlib
    import hmac as _hmac
    import json as _json
    import os as _os
    import re as _re
    src = KERNEL.read_text()
    tree = ast.parse(src)
    wanted = {n.name: n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in names}
    assert set(wanted) == set(names), f"missing: {set(names) - set(wanted)}"
    mod = ast.Module(body=list(wanted.values()), type_ignores=[])
    ns = {"os": _os, "json": _json, "re": _re, "hashlib": _hashlib,
          "hmac": _hmac, "Path": Path, "CFG": {}, "_TUNNEL_TOKEN": "",
          "log": lambda *a: None, "publish": lambda *a, **k: None}
    exec(compile(mod, "serve_qwen38.py", "exec"), ns)  # noqa: S102 (test sandbox)
    return ns


class DigestVerifyTest(unittest.TestCase):
    def test_good_bytes_verify(self):
        ns = load_kernel_funcs("verify_sha256")
        import hashlib
        data = b"fake-cloudflared-binary" * 4096
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            path = f.name
        try:
            self.assertTrue(ns["verify_sha256"](path, hashlib.sha256(data).hexdigest()))
        finally:
            os.unlink(path)

    def test_tampered_bytes_fail(self):
        ns = load_kernel_funcs("verify_sha256")
        import hashlib
        data = b"fake-cloudflared-binary" * 4096
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            path = f.name
        try:
            good = hashlib.sha256(data).hexdigest()
            with open(path, "r+b") as f:
                f.seek(7)
                f.write(b"X")
            self.assertFalse(ns["verify_sha256"](path, good))
            self.assertFalse(ns["verify_sha256"](path, "0" * 64))
        finally:
            os.unlink(path)


class RedactTest(unittest.TestCase):
    def _ns(self, api_key="sk-testkey1234567890abcdef", token="cf-tunnel-token-xyz"):
        ns = load_kernel_funcs("redact")
        ns["CFG"] = {"api_key": api_key}
        ns["_TUNNEL_TOKEN"] = token
        return ns, api_key, token

    def test_literal_secrets_scrubbed(self):
        ns, api_key, token = self._ns()
        lines = [
            f"using api key {api_key} to serve",
            f"env TUNNEL_TOKEN value {token} leaked?",
            f"Authorization: Bearer {api_key}",
        ]
        for line in lines:
            red = ns["redact"](line)
            self.assertNotIn(api_key, red)
            self.assertNotIn(token, red)
            self.assertIn("[REDACTED]", red)

    def test_generic_shapes_caught_without_config(self):
        ns = load_kernel_funcs("redact")
        ns["CFG"] = {}
        ns["_TUNNEL_TOKEN"] = ""
        self.assertIn("[REDACTED]", ns["redact"]("key is sk-abcdefghij1234567890 done"))
        red = ns["redact"]('send -H "Authorization: Bearer abcDEF123" now')
        self.assertNotIn("abcDEF123", red)
        red = ns["redact"]("connector token=supersecretvalue exiting")
        self.assertNotIn("supersecretvalue", red)

    def test_benign_lines_untouched(self):
        ns, _, _ = self._ns()
        line = "   verified official cloudflared 2026.9.1"
        self.assertEqual(ns["redact"](line), line)


class AtomicStateTest(unittest.TestCase):
    def test_0600_under_restrictive_umask(self):
        old = os.umask(0o077)
        try:
            with tempfile.TemporaryDirectory() as td:
                target = Path(td) / "state.json"
                with mock.patch.object(launch, "STATE_FILE", target):
                    launch.write_state({"api_key": "sk-secret", "topic": "t"})
                mode = stat.S_IMODE(os.stat(target).st_mode)
                self.assertEqual(mode, 0o600, oct(mode))
                self.assertEqual(json.loads(target.read_text())["api_key"], "sk-secret")
                leftovers = [p for p in Path(td).iterdir() if p.name != "state.json"]
                self.assertEqual(leftovers, [])
        finally:
            os.umask(old)

    def test_overwrite_keeps_0600(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "state.json"
            target.write_text("{}")
            os.chmod(target, 0o644)
            with mock.patch.object(launch, "STATE_FILE", target):
                launch.write_state({"api_key": "sk-rotated"})
            self.assertEqual(stat.S_IMODE(os.stat(target).st_mode), 0o600)
            self.assertEqual(json.loads(target.read_text())["api_key"], "sk-rotated")


class PinHygieneTest(unittest.TestCase):
    def test_no_latest_url_and_pinned_digest(self):
        src = KERNEL.read_text()
        self.assertNotIn("latest/download", src)
        self.assertNotIn("releases/latest", src)
        m = re.search(r'CLOUDFLARED_SHA256 = "([0-9a-f]{64})"', src)
        self.assertIsNotNone(m, "pinned 64-hex digest missing")
        self.assertIn("CLOUDFLARED_VERSION", src)
        self.assertIn("2026-09-24", src)  # digest verification date recorded

    def test_host_loopback_and_ready_without_key(self):
        src = KERNEL.read_text()
        self.assertIn('"--host", "127.0.0.1"', src)
        m = re.search(r'publish\("ready",.*?startup_secs=startup\)', src, re.S)
        self.assertIsNotNone(m)
        self.assertNotIn("api_key", m.group(0))


class LauncherHygieneTest(unittest.TestCase):
    def test_ready_renders_local_key_not_event_key(self):
        ev = {"phase": "ready", "endpoint": "https://x.trycloudflare.com/v1",
              "api_key": "REMOTE-UNTRUSTED", "model": "qwen3.8-27b",
              "max_model_len": 262144}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            launch.render_event(ev, api_key="LOCAL-KEY")
        out = buf.getvalue()
        self.assertIn("LOCAL-KEY", out)
        self.assertNotIn("REMOTE-UNTRUSTED", out)

    def test_ready_without_local_key_does_not_crash(self):
        ev = {"phase": "ready", "endpoint": "https://x.trycloudflare.com/v1",
              "model": "qwen3.8-27b"}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            launch.render_event(ev, api_key=None)
        self.assertIn("YOUR ENDPOINT IS LIVE", buf.getvalue())

    def test_api_key_hint(self):
        self.assertEqual(launch.api_key_hint({"api_key": "sk-local"}), "sk-local")
        self.assertIn("KTL_API_KEY", launch.api_key_hint(
            {"api_key": "", "api_key_secret": "KTL_API_KEY"}))
        self.assertTrue(launch.api_key_hint({"api_key": ""}))

    def test_ntfy_read_cap_and_kaggle_resolution(self):
        src = (REPO / "launch.py").read_text()
        self.assertIn("r.read(1024 * 1024)", src)
        self.assertIn('shutil.which("kaggle")', src)
        self.assertIn("0o600", src)


if __name__ == "__main__":
    unittest.main()
