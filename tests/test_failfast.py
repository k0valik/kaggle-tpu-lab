"""Stage C: unit tests for fail-fast gates and death-report diagnostics.

Runnable without Kaggle/TPU. Kernel helpers are extracted from the shipped
`serve_qwen38.py` source via ast so the tests track the real code.
"""
import ast
import collections
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_kernel_funcs(*names, **extra):
    """Exec the named top-level functions from serve_qwen38.py in a sandbox."""
    src = (REPO / "qwen38-27b" / "kernel" / "serve_qwen38.py").read_text()
    tree = ast.parse(src)
    wanted = {n.name: n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in names}
    assert set(wanted) == set(names), f"missing: {set(names) - set(wanted)}"
    mod = ast.Module(body=list(wanted.values()), type_ignores=[])
    logs, events = [], []

    def log(msg):
        logs.append(msg)

    def publish(phase, **kw):
        events.append((phase, kw))

    ns = {"os": os, "re": re, "sys": sys, "collections": collections,
          "Path": Path, "CFG": {}, "log": log, "publish": publish}
    ns.update(extra)
    exec(compile(mod, "serve_qwen38.py", "exec"), ns)  # noqa: S102 (test sandbox)
    return ns, logs, events


class SanitizeTpuEnvTest(unittest.TestCase):
    def test_drops_poisoned_vars(self):
        ns, logs, _ = load_kernel_funcs("sanitize_tpu_env")
        os.environ["TPU_WORKER_HOSTNAMES"] = "WARNING: could not determine something"
        os.environ["TPU_WORKER_ADDRS"] = "10.0.0.1"
        try:
            ns["sanitize_tpu_env"]()
        finally:
            os.environ.pop("TPU_WORKER_HOSTNAMES", None)
            os.environ.pop("TPU_WORKER_ADDRS", None)
        self.assertNotIn("TPU_WORKER_HOSTNAMES", os.environ)
        self.assertTrue(any("TPU_WORKER_HOSTNAMES" in m for m in logs))

    def test_clean_env_is_quiet(self):
        ns, logs, _ = load_kernel_funcs("sanitize_tpu_env")
        for k in ("TPU_WORKER_HOSTNAMES", "TPU_WORKER_ADDRS"):
            os.environ.pop(k, None)
        ns["sanitize_tpu_env"]()
        self.assertEqual(logs, [])


class DeathReportOffsetTest(unittest.TestCase):
    def _logfile(self, old, new):
        p = Path(tempfile.mkdtemp()) / "vllm.log"
        p.write_bytes(old + new)
        return p, len(old)

    def test_ignores_previous_run(self):
        old = b"[vllm] Traceback (most recent call last)\n[vllm] RuntimeError: old boom\n"
        new = b"[vllm] (EngineCore) INFO startup ok\n"
        p, off = self._logfile(old, new)
        ns, _, _ = load_kernel_funcs(
            "server_death_report", RAW_LOG=p, _LOG_START=off)
        cause, block, _ = ns["server_death_report"]()
        self.assertNotIn("old boom", cause + block)

    def test_sees_current_run(self):
        old = b"[vllm] (EngineCore) INFO previous run fine\n"
        new = (b"[vllm] Traceback (most recent call last)\n"
               b"[vllm] RuntimeError: new boom\n")
        p, off = self._logfile(old, new)
        ns, _, _ = load_kernel_funcs(
            "server_death_report", RAW_LOG=p, _LOG_START=off)
        cause, block, _ = ns["server_death_report"]()
        self.assertIn("new boom", cause + block)


class ServerDiedHintsTest(unittest.TestCase):
    def _run(self, returncode, report):
        p = Path(tempfile.mkdtemp()) / "vllm.log"
        p.write_bytes(b"")
        ns, logs, events = load_kernel_funcs(
            "server_died", "server_death_report", RAW_LOG=p, _LOG_START=0)
        ns["server_death_report"] = lambda *a, **k: report
        server = type("S", (), {"returncode": returncode,
                                "tail": collections.deque(["line1"],
                                                          maxlen=200)})()
        with self.assertRaises(SystemExit):
            ns["server_died"](server, "stopped")
        return logs, events

    def test_rc0_means_external_stop(self):
        logs, _ = self._run(0, ("", "", ""))
        self.assertTrue(any("Save & Run All" in m for m in logs))

    def test_unknown_cause_points_at_log(self):
        logs, events = self._run(1, ("", "", ""))
        self.assertTrue(any("vllm.log" in m for m in logs))
        phase, kw = events[0]
        self.assertEqual(phase, "stopped")
        self.assertEqual(kw["rc"], 1)

    def test_known_hint_passes_through(self):
        logs, _ = self._run(1, ("some cause", "block", "known hint"))
        self.assertTrue(any("known hint" in m for m in logs))


if __name__ == "__main__":
    unittest.main()
