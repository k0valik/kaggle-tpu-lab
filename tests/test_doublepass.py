"""Double-pass gap tests: config-compat, MTP-head fallback, tar-slip guard,
BUILD_CONFIGS pin, fail-hard propagation, push secrecy.

Runnable without Kaggle/TPU. Kernel helpers are extracted from the shipped
`serve_qwen38.py` source via ast so the tests track the real code.
"""
import ast
import io
import os
import re
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

KERNEL = (REPO / "qwen38-27b" / "kernel" / "serve_qwen38.py").read_text()


def load_kernel_funcs(*names, **extra):
    import json as _json
    import os as _os
    tree = ast.parse(KERNEL)
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
    ns.update(extra)
    # Module-level constants the helpers close over (tracked from source).
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                getattr(node.targets[0], "id", "") == "CONFIG_SHAPE_KEYS":
            ns["CONFIG_SHAPE_KEYS"] = ast.literal_eval(node.value)
    exec(compile(mod, "serve_qwen38.py", "exec"), ns)  # noqa: S102 (test sandbox)
    return ns, logs, events


BASE_CFG = {"hidden_size": 5120, "num_hidden_layers": 64,
            "num_attention_heads": 40, "num_key_value_heads": 8,
            "intermediate_size": 13824, "vocab_size": 202548,
            "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}


class ConfigCompatTest(unittest.TestCase):
    def test_identical(self):
        ns, _, _ = load_kernel_funcs("configs_compatible")
        self.assertTrue(ns["configs_compatible"](dict(BASE_CFG), dict(BASE_CFG)))

    def test_reshaped_head_diverges(self):
        ns, _, _ = load_kernel_funcs("configs_compatible")
        other = dict(BASE_CFG, intermediate_size=8192)
        self.assertFalse(ns["configs_compatible"](other, BASE_CFG))

    def test_missing_key_diverges(self):
        ns, _, _ = load_kernel_funcs("configs_compatible")
        other = {k: v for k, v in BASE_CFG.items() if k != "vocab_size"}
        self.assertFalse(ns["configs_compatible"](other, BASE_CFG))

    def test_tokenizer_noise_ignored(self):
        ns, _, _ = load_kernel_funcs("configs_compatible")
        other = dict(BASE_CFG, chat_template="jinja", revision="fp8")
        self.assertTrue(ns["configs_compatible"](other, BASE_CFG))


class MtpHeadTest(unittest.TestCase):
    def _index(self, weight_map):
        d = Path(tempfile.mkdtemp())
        (d / "model.safetensors.index.json").write_text(
            __import__("json").dumps({"weight_map": weight_map}))
        return str(d)

    def test_head_present(self):
        ns, _, _ = load_kernel_funcs("has_mtp_head")
        d = self._index({"blk.0": "a.safetensors",
                         "mtp.draft": "b.safetensors"})
        self.assertTrue(ns["has_mtp_head"](d))

    def test_head_missing(self):
        ns, _, _ = load_kernel_funcs("has_mtp_head")
        d = self._index({"blk.0": "a.safetensors"})
        self.assertFalse(ns["has_mtp_head"](d))

    def test_broken_index_is_fail_safe(self):
        ns, _, _ = load_kernel_funcs("has_mtp_head")
        self.assertFalse(ns["has_mtp_head"]("/nonexistent-dir-xyz"))


class TarSlipTest(unittest.TestCase):
    def _tar(self, names):
        p = Path(tempfile.mkdtemp()) / "c.tar"
        with tarfile.open(p, "w") as tf:
            for n in names:
                ti = tarfile.TarInfo(n)
                ti.size = 0
                tf.addfile(ti, io.BytesIO(b""))
        return str(p)

    def test_clean_tar(self):
        ns, _, _ = load_kernel_funcs("safe_tar_members")
        self.assertEqual(ns["safe_tar_members"](self._tar(["xla_cache/0", "a/b"])), 2)

    def test_dotdot_rejected(self):
        ns, _, _ = load_kernel_funcs("safe_tar_members")
        with self.assertRaises(ValueError):
            ns["safe_tar_members"](self._tar(["ok", "../evil"]))

    def test_absolute_rejected(self):
        ns, _, _ = load_kernel_funcs("safe_tar_members")
        with self.assertRaises(ValueError):
            ns["safe_tar_members"](self._tar(["/tmp/evil"]))


class BuildConfigsPinTest(unittest.TestCase):
    EXPECTED = {(262144, 4, 3, False), (131072, 16, 3, False),
                (262144, 4, 3, True)}

    def _configs(self):
        tree = ast.parse(KERNEL)
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BUILD_CONFIGS":
                return {tuple(sorted(d.items())) for d in ast.literal_eval(node.value)}
        raise AssertionError("BUILD_CONFIGS not found")

    def test_entries(self):
        got = {(d["max_model_len"], d["max_num_seqs"], d["mtp_tokens"], d["text_only"])
               for d in ({k: v for k, v in t} for t in self._configs())}
        self.assertEqual(got, self.EXPECTED)

    def test_serve_defaults_covered(self):
        # Pin serve-argparse defaults (read from source: main() would execute
        # on call) against BUILD_CONFIGS: changing a default must force a
        # BUILD_CONFIGS review.
        src = (REPO / "launch.py").read_text()
        mlen = int(re.search(r'"--max-model-len".*?default=(\d+)', src, re.S).group(1))
        mseq = int(re.search(r'"--max-num-seqs".*?default=(\d+)', src, re.S).group(1))
        mtp = int(re.search(r'"--mtp".*?default=(\d+)', src, re.S).group(1))
        self.assertIn((mlen, mseq, mtp, False), self.EXPECTED)

    def test_mtp_zero_documents_cold(self):
        # The sync comment above BUILD_CONFIGS states the --mtp 0 policy.
        head = KERNEL.split("BUILD_CONFIGS = [")[0][-600:]
        self.assertIn("mtp", head.lower())
        self.assertIn("cold", head.lower())


class FailHardPropagationTest(unittest.TestCase):
    def test_fetch_raises_instead_of_swallowing(self):
        import urllib.request
        real_urlopen = urllib.request.urlopen

        def boom(*a, **k):
            raise ConnectionError("net down")

        urllib.request.urlopen = boom
        try:
            ns, _, _ = load_kernel_funcs(
                "fetch_cloudflared", "verify_sha256", "_install_verified_binary",
                CLOUDFLARED=str(Path(tempfile.mkdtemp()) / "cf"),
                CLOUDFLARED_URL="https://example.invalid/cf",
                CLOUDFLARED_SHA256="0" * 64, CLOUDFLARED_VERSION="test",
                bundle=None, tempfile=__import__("tempfile"),
                hashlib=__import__("hashlib"), hmac=__import__("hmac"),
                urllib=urllib, time=__import__("time"))
            with self.assertRaises(ConnectionError):
                ns["fetch_cloudflared"]()
        finally:
            urllib.request.urlopen = real_urlopen

    def test_abort_wiring_present(self):
        # The straight-line caller (not a function) must convert fetch
        # failure into publish(failed, step=tunnel-binary) + exit. Find the
        # call site (a bare call, not the `def` line).
        calls = [m.start() for m in re.finditer(r"^[ \t]*fetch_cloudflared\(\)", KERNEL, re.M)]
        calls = [i for i in calls if "def " not in KERNEL[max(0, i - 4):i + len("fetch_cloudflared")]]
        self.assertTrue(calls, "no fetch_cloudflared() call site found")
        window = KERNEL[max(0, calls[0] - 400):calls[0] + 600]
        self.assertIn("tunnel-binary", window)
        self.assertIn("sys.exit(1)", window)


class PushSecrecyTest(unittest.TestCase):
    SRC = (REPO / "launch.py").read_text()

    def test_metadata_private_and_keyless(self):
        # The metadata DICT (following each kernel-metadata.json write) must
        # be private and carry no secrets. The CFG dict built earlier in the
        # same function legitimately holds api_key (it is injected into the
        # kernel source, never into metadata) — so only inspect what follows
        # the marker.
        markers = list(re.finditer(r"kernel-metadata\.json", self.SRC))
        self.assertTrue(markers, "no metadata writes found")
        for m in markers:
            end = self.SRC.find("}, indent=1)", m.start())
            self.assertTrue(end > m.start(), "metadata dict end not found")
            window = self.SRC[m.start():end]
            self.assertIn('"is_private"', window)
            self.assertNotIn("api_key", window)
            self.assertNotIn("TUNNEL_TOKEN", window)
            self.assertNotIn("HF_TOKEN", window.replace("token-secret", ""))


if __name__ == "__main__":
    unittest.main()
