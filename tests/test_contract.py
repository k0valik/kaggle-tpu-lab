"""Stage E: pin the frozen Tauri companion contract.

The companion app (vendored `src/`, `src-tauri/`) is frozen — future stages
must not silently break what it depends on (see AGENTS.md):

1. `launch.py serve` argv: the 10 app-known flags assembled by the app's
   `serve_args()` (`src-tauri/src/state/settings.rs`). Added launcher flags
   are fine; removing one of these breaks Start.
2. State-file keys written by `cmd_serve`: the app's `LocalState::read`
   (`src-tauri/src/kaggle/launcher.rs`) reads kernel/topic/api_key/model/
   submitted_at/keepalive_min. `api_key_secret` is a known-optional extra
   (only present with `--api-key-secret`); anything else is a contract break.
3. ntfy phase names: the app's `apply_event` (`src-tauri/src/state/
   machine.rs`) handles the KNOWN set below. Added phases are allowed
   (removals fail). Phases the app deliberately ignores (image-test*,
   probe-*, build-config-done, bundle-built) are NOT pinned — the kernel
   may drop them freely.

Method: argparse introspection (the serve block's `add_argument` calls) +
source inspection via `ast`/regex. No subprocess, no Kaggle/TPU.
Same style as `tests/test_weights_source.py` (`launch` imports cleanly —
`main()` is behind the `__main__` guard).
"""
import ast
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import launch  # noqa: E402  (safe: main() is behind __main__ guard)

LAUNCH_SRC = (REPO / "launch.py").read_text()
SETTINGS_RS = (REPO / "src-tauri" / "src" / "state" / "settings.rs").read_text()
LAUNCHER_RS = (REPO / "src-tauri" / "src" / "kaggle" / "launcher.rs").read_text()

# The 10 flags the frozen app passes to `launch.py serve` (settings.rs serve_args).
APP_KNOWN_FLAGS = frozenset([
    "--model", "--keepalive-min",
    "--max-model-len", "--mtp", "--reasoning-effort",
    "--fast-start", "--text-only", "--no-async-scheduling",
    "--max-len", "--streams",
])

# State keys the frozen app reads (launcher.rs LocalState::read + state_model).
APP_STATE_KEYS = frozenset(
    ["kernel", "topic", "api_key", "model", "submitted_at", "keepalive_min"])
OPTIONAL_STATE_KEYS = frozenset(["api_key_secret"])

# Phases the frozen app handles (machine.rs apply_event match arms).
APP_KNOWN_PHASES = frozenset([
    "install", "installed",
    "mtp-patch-applied", "mtp-patch-failed",
    "cache-restored", "cache-missing", "cache-bypassed",
    "weights-mounted", "weights-download", "weights-progress", "weights-downloaded",
    "server-launch", "loading", "loaded", "warmed",
    "tunnel-url", "tunnel-reconnecting", "tunnel-failed",
    "compiling", "serving", "benchmark", "benchmark-error",
    "ready", "heartbeat", "failed", "auto-shutdown", "stopped",
])

KERNELS = [REPO / "qwen38-27b" / "kernel" / "serve_qwen38.py",
           REPO / "glm53-flash" / "kernel" / "serve_glm53.py"]


def serve_block_flags():
    """Option strings declared in the `serve` subparser block of launch.py.

    `main()` owns the argparse builder (no importable factory), so this
    statically extracts the serve block's `add_argument("--...")` calls —
    argparse introspection without spawning a subprocess.
    """
    lines = LAUNCH_SRC.splitlines()
    start = next(i for i, l in enumerate(lines) if 'add_parser("serve"' in l)
    end = next(i for i in range(start + 1, len(lines)) if "add_parser(" in lines[i])
    block = "\n".join(lines[start:end])
    flags = set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', block))
    assert flags, "no serve flags extracted — launch.py serve block moved?"
    return flags


def app_serve_flags():
    """`--...` literals the app assembles for `launch.py serve`."""
    flags = set(re.findall(r'"(--[a-z][a-z-]*)"', SETTINGS_RS))
    assert APP_KNOWN_FLAGS <= flags, \
        f"settings.rs drifted from test's known set: {sorted(APP_KNOWN_FLAGS - flags)}"
    return flags


def serve_state_keys():
    """Keys of the `state = {...}` dict written by `cmd_serve`."""
    tree = ast.parse(LAUNCH_SRC)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "cmd_serve")
    dicts = [n.value for n in ast.walk(fn)
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "state" for t in n.targets)
             and isinstance(n.value, ast.Dict)]
    assert len(dicts) == 1, f"expected one state dict in cmd_serve, found {len(dicts)}"
    keys = {k.value for k in dicts[0].keys if isinstance(k, ast.Constant)}
    assert keys, "no state keys extracted — cmd_serve state dict moved?"
    return keys


def app_state_keys():
    """`v.get("...")` literals in the app's state-file reader."""
    m = re.search(r"impl LocalState(.*?)fn state_model", LAUNCHER_RS, re.S)
    assert m, "launcher.rs LocalState block moved?"
    keys = set(re.findall(r'\.get\("([a-z_]+)"\)', m.group(1)))
    keys.add("model")  # read via state_model(), not a direct v.get in read()
    assert APP_STATE_KEYS <= keys, \
        f"launcher.rs drifted from test's known set: {sorted(APP_STATE_KEYS - keys)}"
    return keys


def published_phases():
    """All ntfy phases the repo can emit: PHASE_TEXT + launcher branches +
    kernel `publish("...")` first args."""
    phases = set(launch.PHASE_TEXT)
    phases.update(re.findall(r'phase"?\s*==\s*"([^"]+)"', LAUNCH_SRC))
    for m in re.findall(r'get\("phase"\) in \(([^)]*)\)', LAUNCH_SRC):
        phases.update(re.findall(r'"([^"]+)"', m))
    for k in KERNELS:
        tree = ast.parse(k.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "publish"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)):
                phases.add(node.args[0].value)
    assert phases, "no phases extracted"
    return phases


class ServeArgvContractTest(unittest.TestCase):
    def test_app_flags_still_accepted(self):
        self.assertTrue(APP_KNOWN_FLAGS <= serve_block_flags(),
                        f"serve dropped app-known flags: {sorted(APP_KNOWN_FLAGS - serve_block_flags())}")

    def test_app_uses_nothing_unknown(self):
        self.assertTrue(app_serve_flags() <= serve_block_flags(),
                        f"app passes flags serve does not know: {sorted(app_serve_flags() - serve_block_flags())}")


class StateKeyContractTest(unittest.TestCase):
    def test_serve_writes_app_keys(self):
        self.assertTrue(APP_STATE_KEYS <= serve_state_keys(),
                        f"cmd_serve dropped state keys: {sorted(APP_STATE_KEYS - serve_state_keys())}")

    def test_no_unexpected_state_keys(self):
        self.assertTrue(serve_state_keys() <= APP_STATE_KEYS | OPTIONAL_STATE_KEYS,
                        f"new state keys need app support: {sorted(serve_state_keys() - APP_STATE_KEYS - OPTIONAL_STATE_KEYS)}")

    def test_app_reads_only_written_keys(self):
        self.assertTrue(app_state_keys() <= serve_state_keys() | OPTIONAL_STATE_KEYS,
                        f"app reads keys serve never writes: {sorted(app_state_keys() - serve_state_keys() - OPTIONAL_STATE_KEYS)}")


class NtfyPhaseContractTest(unittest.TestCase):
    def test_known_phases_still_published(self):
        actual = published_phases()
        self.assertTrue(APP_KNOWN_PHASES <= actual,
                        f"removed ntfy phases break the app: {sorted(APP_KNOWN_PHASES - actual)}")


if __name__ == "__main__":
    unittest.main()
