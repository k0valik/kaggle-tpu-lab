"""Stage B: unit tests for the generic weights-source plumbing.

Runnable without Kaggle/TPU. `launch.py` imports cleanly (main guard);
kernel helpers are extracted from the shipped `serve_qwen38.py` source
via ast so the tests track the real code, not a copy.
"""
import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import launch  # noqa: E402  (safe: main() is behind __main__ guard)


def load_kernel_funcs(*names):
    """Exec the named top-level functions from serve_qwen38.py in a sandbox."""
    import json as _json
    import os as _os
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

    ns = {"os": _os, "json": _json, "Path": Path, "CFG": {},
          "log": log, "publish": publish}
    exec(compile(mod, "serve_qwen38.py", "exec"), ns)  # noqa: S102 (test sandbox)
    return ns, logs, events


class OptionalDatasetTest(unittest.TestCase):
    def test_empties(self):
        for v in ("", "none", "None", "NONE", "  none  ", "null", "-", None):
            self.assertEqual(launch.optional_dataset(v), "", repr(v))

    def test_passthrough(self):
        self.assertEqual(launch.optional_dataset("rahim3/qwen3-8-27b-bf16"),
                         "rahim3/qwen3-8-27b-bf16")
        self.assertEqual(launch.optional_dataset("user/my-fp8"), "user/my-fp8")

    def test_phase_text_covers_new_phases(self):
        for phase in ("weights-progress", "cache-bypassed"):
            self.assertIn(phase, launch.PHASE_TEXT)


class WeightsEmptyTest(unittest.TestCase):
    def test(self):
        ns, _, _ = load_kernel_funcs("_weights_dataset_is_empty")
        f = ns["_weights_dataset_is_empty"]
        self.assertTrue(f(""))
        self.assertTrue(f("none"))
        self.assertTrue(f("None"))
        self.assertTrue(f(None))
        self.assertFalse(f("rahim3/qwen3-8-27b-bf16"))
        self.assertFalse(f("user/my-fp8"))


class ChatTemplateTest(unittest.TestCase):
    def _dir(self, files):
        d = Path(tempfile.mkdtemp())
        for name, content in files.items():
            (d / name).write_text(content)
        return d

    def test_v4_config_wins(self):
        ns, _, _ = load_kernel_funcs("chat_template")
        d = self._dir({
            "tokenizer_config.json": json.dumps({"chat_template": "v4-template"}),
            "chat_template.jinja": "v5-template",
        })
        self.assertEqual(ns["chat_template"](str(d)), "v4-template")

    def test_v5_jinja_fallback(self):
        ns, _, _ = load_kernel_funcs("chat_template")
        d = self._dir({
            "tokenizer_config.json": json.dumps({"bos_token": "<s>"}),
            "chat_template.jinja": "v5-template",
        })
        self.assertEqual(ns["chat_template"](str(d)), "v5-template")

    def test_missing_returns_none(self):
        ns, _, _ = load_kernel_funcs("chat_template")
        d = self._dir({"tokenizer_config.json": json.dumps({})})
        self.assertIsNone(ns["chat_template"](str(d)))


class HfTokenTest(unittest.TestCase):
    def test_env_lookup(self):
        ns, _, _ = load_kernel_funcs("hf_token")
        os.environ["HF_TOKEN"] = "  hf_test_123  "
        try:
            self.assertEqual(ns["hf_token"](), "hf_test_123")
        finally:
            del os.environ["HF_TOKEN"]

    def test_empty_without_secret(self):
        ns, _, _ = load_kernel_funcs("hf_token")
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            os.environ.pop(k, None)
        # kaggle_secrets is not installed here -> best-effort path returns ""
        self.assertEqual(ns["hf_token"](), "")


class SnapshotValidationTest(unittest.TestCase):
    def _repo(self, weight_map):
        d = Path(tempfile.mkdtemp())
        shards = sorted(set(weight_map.values()))
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}))
        for s in shards:
            (d / s).write_bytes(b"x" * 1024)
        return d

    def test_valid_repo(self):
        ns, _, _ = load_kernel_funcs("complete_bf16_repo")
        d = self._repo({"blk.0.attn": "shard-00001.safetensors",
                        "mtp.head": "shard-00002.safetensors"})
        ok, detail = ns["complete_bf16_repo"](str(d))
        self.assertTrue(ok, detail)
        self.assertIn("2 shards", detail)

    def test_missing_shard(self):
        ns, _, _ = load_kernel_funcs("complete_bf16_repo")
        d = Path(tempfile.mkdtemp())
        (d / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "shard-00001.safetensors"}}))
        ok, detail = ns["complete_bf16_repo"](str(d))
        self.assertFalse(ok)
        self.assertIn("missing", detail)

    def test_invalid_index(self):
        ns, _, _ = load_kernel_funcs("complete_bf16_repo")
        d = Path(tempfile.mkdtemp())
        ok, _ = ns["complete_bf16_repo"](str(d))
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
