"""
Serve Qwen3.8-27B (bf16) on a Kaggle TPU v5e-8 with vLLM.

This script is pushed to Kaggle as a script kernel by ../launch.py, which fills
in the CFG line below. It also runs standalone with defaults (e.g. pasted into
a Kaggle notebook/script in the UI) — then it just prints instead of using ntfy.

The script verifies a pinned official Cloudflare binary, installs vllm-tpu,
loads the model, compiles its own TPU graphs and serves through a tunnel.
Credentials are kept out of ntfy; progress is allowlisted and signed. Dataset
executables and compiled caches are disabled. Startup timings for this hardened
configuration have not yet been measured. See SECURITY.md for remaining trust.
"""
import base64
import collections
import struct
import zlib
import glob
import gzip
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# BEGIN EMBEDDED HARDENING
"""Standard-library security helpers, also embedded in the standalone TPU script."""
import hashlib
import hmac
import json
import math
import os
import re
import tempfile
import time
import urllib.request
from pathlib import Path

# https://github.com/cloudflare/cloudflared/releases/tag/2026.8.3
# Update the version and digest together after checking the official release.
CLOUDFLARED_VERSION = "2026.8.3"
CLOUDFLARED_SHA256 = "f29324fe934d1e100617484c78deef803c4dc2cd351d645bbde42e96b4fccc5e"
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/download/"
    f"{CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
)


def download_cloudflared(destination):
    """Download to a private temporary file; install only after verification.

    Never trust an existing file or a dataset copy. Call in a private directory.
    A failed download or checksum raises, preventing tunnel startup.
    """
    destination = Path(destination)
    fd, tmp = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as out:
            with urllib.request.urlopen(CLOUDFLARED_URL, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    out.write(chunk)
            if not hmac.compare_digest(digest.hexdigest(), CLOUDFLARED_SHA256):
                raise RuntimeError("cloudflared SHA-256 mismatch; refusing to execute")
            out.flush()
            os.fsync(out.fileno())
        os.chmod(tmp, 0o700)
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def public_event(phase, extra):
    """Only allow structured progress metadata, never arbitrary logs or secrets."""
    if not isinstance(phase, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", phase):
        raise ValueError("Invalid progress phase")
    result = {"phase": phase}
    for field in ("elapsed_s", "startup_secs", "decode_tok_s", "max_model_len",
                  "max_num_seqs", "keepalive_min", "up_min", "served_min",
                  "secs", "entries", "rc"):
        value = extra.get(field)
        if type(value) in (int, float) and math.isfinite(value):
            result[field] = value
    if type(extra.get("covers_this_config")) is bool:
        result["covers_this_config"] = extra["covers_this_config"]
    endpoint = extra.get("endpoint")
    if isinstance(endpoint, str) and re.fullmatch(
            r"https://[a-z0-9-]+\.trycloudflare\.com/v1", endpoint):
        result["endpoint"] = endpoint
    # Known values only: arbitrary strings can contain credentials or prompt text.
    if extra.get("model") == "qwen3.8-27b":
        result["model"] = "qwen3.8-27b"
    if extra.get("step") in ("install", "server", "health-timeout", "tunnel"):
        result["step"] = extra["step"]
    return result


def sign_event(event, topic, event_key):
    if not event_key:
        raise ValueError("Missing progress signing key")
    payload = json.dumps({"topic": topic, "time": int(time.time()), "event": event},
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    signature = hmac.new(event_key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return json.dumps({"payload": payload, "signature": signature})


def verify_event(message, topic, event_key):
    """Authenticate progress before displaying it; legacy unsigned messages fail closed."""
    if not event_key or not isinstance(message, str) or len(message) > 16384:
        return None
    try:
        envelope = json.loads(message)
        payload, signature = envelope["payload"], envelope["signature"]
        if not isinstance(payload, str) or not isinstance(signature, str):
            return None
        expected = hmac.new(event_key.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return None
        data = json.loads(payload)
        stamp = data["time"]
        if type(stamp) is not int or not time.time() - 86400 <= stamp <= time.time() + 60:
            return None
        if data["topic"] != topic or not isinstance(data["event"], dict):
            return None
        event = data["event"]
        return stamp, public_event(event["phase"], event)
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError, RecursionError):
        return None


def write_private_state(path, state):
    """Atomically replace state with a mode-0600 file, without following symlinks."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".ktl-state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(state, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

# END EMBEDDED HARDENING

CFG = None  # __LAUNCHER_CONFIG__  (launch.py replaces this line)

DEFAULTS = {
    "vllm_tpu_version": "0.28.0",
    "weights_dataset": "rahim3/qwen3-8-27b-bf16",     # HF mirror of Qwen/Qwen3.8-27B
    "hf_model_id": "Qwen/Qwen3.8-27B",                # fallback download source
    "max_model_len": 262144,       # native context; drop to 131072 + max_num_seqs 16 for throughput
    "max_num_seqs": 4,
    "mtp_tokens": 3,               # MTP spec decoding (+34% in our A/B test). Stock vllm-tpu
                                   # 0.28.0 corrupts outputs with it (missing GDN state
                                   # rollback); we apply patches/mtp-rollback-v0280.diff
                                   # (a port of upstream PR #3178) before serving —
                                   # verified lossless, 12/12 greedy exact-match.
    "reasoning_effort_default": "xhigh",   # server-side default: xhigh | medium | low
    "tool_call_parser": "qwen3_coder",  # matches Qwen3.8's XML tool format; "" disables
    "text_only": False,            # True: skip the vision tower + its TPU graphs (saves ~8 min,
                                   # image inputs then error out)
    "min_token_bucket": 64,        # smallest padded batch (tokens); 16 = more graphs to compile
    "precompile_workers": 4,       # parallel XLA compile threads (1 = sequential)
    "fast_start": False,           # Skip initial precompile; first-use shapes compile cold
    "keepalive_min": 480,          # auto-shutdown guard (Kaggle TPU caps at 9h anyway)
    "api_key": "",                 # generated if empty
    "ntfy_topic": "",              # optional: signed, allowlisted progress only
    "event_key": "",               # separate HMAC key; never published
    "served_model_name": "qwen3.8-27b",
    "verbose": False,              # show every vLLM log line (always saved to vllm.log)
    "build_bundle": False,         # maintainer mode: build the env dataset instead of serving
}
CFG = {**DEFAULTS, **(CFG or {})}
# Notebook flow: drop overrides in a serve_config.json next to this script.
_cfg_file = Path("serve_config.json")
if _cfg_file.exists():
    CFG.update(json.loads(_cfg_file.read_text()))
if not CFG["api_key"]:
    CFG["api_key"] = "sk-" + secrets.token_hex(16)

PORT = 8000
_session = tempfile.TemporaryDirectory(prefix="ktl-")
SESSION_DIR = Path(_session.name)
VENV = str(SESSION_DIR / "venv")
PY = f"{VENV}/bin/python"
XLA_CACHE = str(SESSION_DIR / "xla_cache")
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
RAW_LOG = WORK / "vllm.log"          # every line vLLM/pip print, for debugging
CLOUDFLARED = SESSION_DIR / "cloudflared"
KEY_FILE = SESSION_DIR / "inference-key.json"
write_private_state(KEY_FILE, {"api_key": CFG["api_key"]})
T0 = time.time()
PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}"

os.environ["HF_HOME"] = "/tmp/hf"                 # /kaggle/working is only ~21 GB
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["VLLM_XLA_CACHE_PATH"] = XLA_CACHE
os.environ["MIN_TOKEN_BUCKET"] = str(CFG["min_token_bucket"])
os.environ["NUM_PRECOMPILE_WORKERS"] = str(CFG["precompile_workers"])
if CFG["fast_start"]:
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
# The venv ships its own libtpu; don't let the image's TPU_LIBRARY_PATH override it.
os.environ.pop("TPU_LIBRARY_PATH", None)

_raw = open(RAW_LOG, "a", buffering=1)


def redact(text):
    for secret in (CFG["api_key"], CFG["event_key"]):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def log(*parts):
    line = redact(time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts))
    print(line, flush=True)
    _raw.write(line + "\n")


def elapsed():
    return f"{int(time.time() - T0) // 60} min {int(time.time() - T0) % 60:02d} s"


def banner(step, title, note=""):
    log("")
    log("=" * 70)
    log(f" STEP {step}/6  {title}" + (f"   ({note})" if note else "") + f"   [{elapsed()} so far]")
    log("=" * 70)


def publish(phase, **extra):
    """Send only signed, allowlisted metadata. Logs, prompts and keys stay off ntfy."""
    event = public_event(phase, extra)
    log("PHASE", phase, json.dumps(event))
    if not CFG["ntfy_topic"]:
        return
    try:
        message = sign_event(event, CFG["ntfy_topic"], CFG["event_key"])
        body = {"topic": CFG["ntfy_topic"], "title": f"kaggle-tpu-lab {phase}",
                "message": message}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        log("(ntfy progress unavailable; inspect the private Kaggle session)")


def sh(cmd, tag, show=None, env=None):
    """Run a command; stream its output to vllm.log (and to the console when
    show/verbose). Returns the exit code."""
    show = CFG["verbose"] if show is None else show
    tail = collections.deque(maxlen=40)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env)
    for line in p.stdout:
        line = redact(line.rstrip())
        if not line:
            continue
        tail.append(line)
        _raw.write(f"[{tag}] {line}\n")
        if show:
            print(f"[{tag}] {line[:400]}", flush=True)
    rc = p.wait()
    if rc != 0 and not show:
        log(f"[{tag}] exited with code {rc}; last lines:")
        for ln in list(tail)[-15:]:
            print("    " + ln[:300], flush=True)
    return rc


def find_input(*patterns):
    """Datasets mount at /kaggle/input/<slug> (UI) or /kaggle/input/datasets/<owner>/<slug> (API push)."""
    for pat in patterns:
        hits = glob.glob(f"/kaggle/input/{pat}") + glob.glob(f"/kaggle/input/datasets/*/{pat}")
        if hits:
            return hits[0]
    return None


def fetch_cloudflared():
    download_cloudflared(CLOUDFLARED)
    log("   verified official cloudflared", CLOUDFLARED_VERSION)


# gzip+base64 of patches/mtp-rollback-v0280.diff; regenerated by tools/embed_patch.py
MTP_PATCH_B64 = "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"  # __EMBEDDED_PATCH__


def apply_mtp_patch():
    """Port of upstream PR #3178 (GDN state rollback on rejected draft tokens).
    Without it ANY speculative decoding corrupts outputs on this model."""
    if not MTP_PATCH_B64:
        return True
    Path("/tmp/mtpfix.diff").write_text(
        gzip.decompress(base64.b64decode(MTP_PATCH_B64)).decode())
    origin = subprocess.check_output(
        [PY, "-c", "import importlib.util as u; print(u.find_spec('tpu_inference').origin)"],
        text=True).strip()
    pkg_root = os.path.dirname(os.path.dirname(origin))
    p = subprocess.run(["patch", "-p1", "-d", pkg_root, "-i", "/tmp/mtpfix.diff",
                        "--no-backup-if-mismatch", "-N"], capture_output=True, text=True)
    _raw.write(redact(p.stdout + p.stderr))
    if p.returncode == 0 or "previously applied" in p.stdout:
        return True
    log(p.stdout[-1500:], p.stderr[-500:])
    return False


def runtime_ok():
    r = subprocess.run([PY, "-c", "import importlib.util as u, jax, torch\n"
                        "assert u.find_spec('vllm') and u.find_spec('tpu_inference')\n"
                        "print(jax.__version__, torch.__version__)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        log(f"   runtime check OK (jax {r.stdout.split()[0]}, torch {r.stdout.split()[1]})")
        return True
    log("   runtime check FAILED:", (r.stderr or r.stdout)[-800:])
    return False


def install_runtime(built=None):
    """Fresh venv with vllm-tpu pinned. CPU torch (what vllm-tpu's own Docker
    image uses) — the default PyPI torch drags in ~3 GB of CUDA libraries that
    a TPU never uses. `built` (a date from the env dataset's manifest) pins the
    dependency resolution to that day so the compile cache keeps matching."""
    ver = CFG["vllm_tpu_version"]
    shutil.rmtree(VENV, ignore_errors=True)
    pin = ["--exclude-newer", f"{built}T23:59:59Z"] if built else []
    log("   building venv with uv" + (f" (packages as of {built})" if built else "") + "...")
    if (sh([sys.executable, "-m", "pip", "install", "-q", "uv"], "pip") == 0
            and sh([sys.executable, "-m", "uv", "venv", VENV, "--python", sys.executable, "-q"], "uv") == 0
            and sh([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
                    "--torch-backend=cpu", *pin, f"vllm-tpu=={ver}"], "uv") == 0):
        return "uv"
    log("   uv failed; falling back to pip (slower)")
    shutil.rmtree(VENV, ignore_errors=True)
    if sh([sys.executable, "-m", "venv", "--without-pip", VENV], "venv") != 0:
        return None
    rc = sh([sys.executable, "-m", "pip", "--python", PY, "install", "-q",
             "--extra-index-url", "https://download.pytorch.org/whl/cpu",
             f"vllm-tpu=={ver}"], "pip")
    return "pip" if rc == 0 else None


# ---------------- 1. runtime ----------------
banner(1, "Python runtime", f"vllm-tpu {CFG['vllm_tpu_version']}")
# Verify the tunnel binary before spending time installing or compiling.
# Failure aborts this run; no dataset or existing-file fallback is permitted.
fetch_cloudflared()

t = time.time()
publish("install", vllm_tpu=CFG["vllm_tpu_version"])
log("   installer output goes to", RAW_LOG)
runtime = install_runtime()
if runtime is None or not runtime_ok():
    publish("failed", step="install")
    sys.exit(1)
publish("installed", secs=int(time.time() - t), via=runtime)
if apply_mtp_patch():
    publish("mtp-patch-applied")
elif CFG["mtp_tokens"] > 0:
    publish("mtp-patch-failed", note="disabling MTP: unsafe without the rollback patch")
    CFG["mtp_tokens"] = 0
log(f"   runtime ready in {int(time.time() - t)} s")

# ---------------- 2. XLA compile cache ----------------
banner(2, "Fresh XLA compile cache")
Path(XLA_CACHE).mkdir(mode=0o700)
n_entries = 0
publish("cache-missing")
log("   dataset compile caches are disabled; TPU graphs will compile in this session")

# ---------------- 3. weights ----------------
banner(3, "Model weights", "55 GB bf16 safetensors")
weights_slug = CFG["weights_dataset"].split("/")[-1]
model_path = find_input(weights_slug)
if model_path and os.path.exists(os.path.join(model_path, "config.json")):
    publish("weights-mounted", path=model_path)
else:
    publish("weights-download", model=CFG["hf_model_id"],
            note="attach the weights dataset to skip this (~5 min parallel download)")
    t = time.time()
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(CFG["hf_model_id"], allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "tokenizer*", "vocab*", "merges*"])
    publish("weights-downloaded", secs=int(time.time() - t))


# ---------------- 4. vLLM server ----------------
NOISE = ("vllm._C", "metadata.google.internal", "Triton is installed", "Transparent hugepages",
         "Pin memory is not supported", "Expect torch.Tensor", "Inductor compilation",
         "cloud_tpu_init.py", "SyntaxWarning", "Compilation of worker", "AOT lower skipped",
         "torch_dtype", "UserWarning", "warnings.warn", "resource_tracker", "Precompile worker0 sample",
         "Precompile worker0 gather", "Precompile worker0 compute_and_gather")


def server_args(cfg):
    args = [PY, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--tensor-parallel-size", "8",
            "--max-model-len", str(cfg["max_model_len"]),
            "--max-num-seqs", str(cfg["max_num_seqs"]),
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "--api-key", cfg["api_key"],
            "--served-model-name", cfg["served_model_name"],
            "--reasoning-parser", "qwen3"]
    if cfg["text_only"]:
        # Qwen3.8 is a vision-language checkpoint; we only serve text. This skips
        # the vision tower and roughly halves the number of TPU graphs to compile.
        args += ["--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if cfg["mtp_tokens"] > 0:
        args += ["--speculative-config",
                 json.dumps({"method": "mtp", "num_speculative_tokens": cfg["mtp_tokens"]})]
    if cfg["tool_call_parser"]:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", cfg["tool_call_parser"]]
    if cfg["reasoning_effort_default"] != "xhigh":
        # The chat template defaults reasoning_effort to 'xhigh'; ship a copy with a
        # different default so the server-side default changes without client changes.
        tc = json.loads(Path(model_path, "tokenizer_config.json").read_text())
        template = tc["chat_template"].replace(
            "reasoning_effort|default('xhigh')",
            f"reasoning_effort|default('{cfg['reasoning_effort_default']}')")
        Path("/tmp/chat_template.jinja").write_text(template)
        args += ["--chat-template", "/tmp/chat_template.jinja"]
    return args


def n_token_graphs():
    n, b = 1, CFG["min_token_bucket"]
    while b < 2048:  # vLLM's default max_num_batched_tokens on TPU
        b *= 2
        n += 1
    return n


def make_translator():
    """Turns vLLM's firehose into a handful of human lines. Everything raw still
    lands in vllm.log."""
    st = {"graph": 0, "loads": 0, "said": set()}
    n_graphs = n_token_graphs()

    def once(key, msg):
        if key not in st["said"]:
            st["said"].add(key)
            log(msg)

    def tr(line):
        if CFG["verbose"]:
            print(f"[vllm] {line[:500]}", flush=True)
            return
        if any(k in line for k in NOISE):
            return
        m = re.search(r"Loading weights took ([\d.]+) seconds", line)
        if m:
            st["loads"] += 1
            if st["loads"] == 1:
                log(f"   weights read from the dataset in {float(m.group(1)):.0f} s")
            return
        m = re.search(r"load model weights from storage to TPU: ([\d.]+)", line)
        if m:
            if st["loads"] <= 1:
                log(f"   weights sharded across the 8 TPU chips ({float(m.group(1)):.0f} s)")
            else:
                log("   MTP draft head loaded")
            return
        m = re.search(r"KV cache size: ([\d,]+) tokens", line)
        if m:
            log(f"   KV cache fits {m.group(1)} tokens")
            return
        if "Precompile all the subgraphs" in line:
            log(f"   compiling TPU graphs — {n_graphs} text graphs"
                + ("" if CFG["text_only"] else ", the same again for image inputs,")
                + " + helpers (~20 s each if cached, ~1 min if not)")
            return
        m = re.search(r"Precompile worker\d+ backbone --> \{'num_tokens': (\d+)", line)
        if m:
            st["graph"] += 1
            log(f"     graph {st['graph']}/{n_graphs}: batches of {m.group(1)} tokens")
            return
        if "embed_multimodal" in line or "input_embeddings_merger" in line:
            once("vision-enc", "     warming the image encoder (~5 min; \"text_only\": true skips it)")
            return
        if "backbone with embeds" in line:
            once("vision", "     compiling image-input graphs (~3 min)")
            return
        m = re.search(r"Warm-up call pass finished in ([\d.]+) \[secs\] over (\d+) tasks", line)
        if m:
            if float(m.group(1)) > 5:
                log(f"     warm-up run of {m.group(2)} graphs done ({float(m.group(1)):.0f} s)")
            return
        if "Precompile" in line and "drafter" in line:
            once("mtp", "     compiling speculative-decoding (MTP) graphs")
            return
        if "Precompile" in line or "Compilation of" in line:
            once("helpers", "     compiling sampler / helper graphs")
            return
        if "Application startup complete" in line:
            return
        if " ERROR " in line or "Traceback" in line or "Error:" in line or "rror(" in line:
            print(time.strftime("[%H:%M:%S] ") + f"   [vllm] {line[:400]}", flush=True)
    return tr


def launch_server(cfg):
    publish("server-launch", max_model_len=cfg["max_model_len"],
            max_num_seqs=cfg["max_num_seqs"], mtp=cfg["mtp_tokens"],
            text_only=cfg["text_only"], min_token_bucket=cfg["min_token_bucket"])
    tail = collections.deque(maxlen=200)
    p = subprocess.Popen(server_args(cfg), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
    tr = make_translator()

    def pump():
        for line in p.stdout:
            line = redact(line.rstrip())
            if line:
                tail.append(line)
                _raw.write(f"[vllm] {line}\n")
                tr(line)
    threading.Thread(target=pump, daemon=True).start()
    p.tail = tail
    return p


def healthy(cfg):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/models",
                                     headers={"Authorization": f"Bearer {cfg['api_key']}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def wait_healthy(server, cfg, expect_min):
    t = time.time()
    while time.time() - t < 5400:
        if server.poll() is not None:
            tail = "\n".join(list(server.tail)[-100:])
            log(f"server exited rc={server.returncode}; last output:\n{tail}")
            log(f"full log: {RAW_LOG}")
            publish("failed", step="server", rc=server.returncode, tail=tail[-2500:])
            sys.exit(1)
        if healthy(cfg):
            return int(time.time() - t)
        el = int(time.time() - t)
        if el and el % 120 < 6:
            publish("compiling", elapsed_s=el)
            log(f"   ... {el // 60} min into startup (typically ~{expect_min} min)")
        time.sleep(5)
    publish("failed", step="health-timeout", tail="\n".join(list(server.tail)[-60:])[-2500:])
    sys.exit(1)


def stop_server(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=30)
    time.sleep(10)  # let the TPU runtime free the chips


def completion(cfg, prompt, max_tokens, stream=False, timeout=900):
    body = {"model": cfg["served_model_name"], "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0.0}
    if stream:
        body.update(stream=True, ignore_eos=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    t0 = time.time(); ttft = None; gen = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            ch = obj.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                gen += 1
            if obj.get("usage"):
                gen = obj["usage"].get("completion_tokens", gen)
    return ttft, time.time() - t0, gen


def test_png(w=256, h=256, rgb=(200, 30, 30)):
    """A solid-colour PNG without PIL, for the image self-test."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def chat(cfg, messages, max_tokens=32, timeout=600):
    body = {"model": cfg["served_model_name"], "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def self_test(cfg):
    """Warms the remaining lazy paths, checks an image request when images are
    enabled, and reports single-stream decode speed."""
    tps = None
    try:
        completion(cfg, "Hello", 8)
        txt = completion(cfg, "The capital of France is", 8)["choices"][0]["text"]
        ttft, total, gen = completion(cfg, "Write a short story about a lighthouse.", 192, stream=True)
        tps = (gen - 1) / (total - ttft) if gen > 1 else 0.0
        publish("benchmark", decode_tok_s=round(tps, 1), sanity=txt.strip()[:60])
        log(f"   self-test: {tps:.1f} tok/s single-stream decode; "
            f"'The capital of France is' -> {txt.strip()[:40]!r}")
    except Exception as e:
        publish("benchmark-error", err=str(e)[:200])
    if not cfg["text_only"]:
        try:
            img = "data:image/png;base64," + base64.b64encode(test_png()).decode()
            ans = chat(cfg, [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img}},
                {"type": "text", "text": "What colour is this image? One word."}]}])
            publish("image-test", answer=ans.strip()[:40])
            log(f"   image request works (a red square -> {ans.strip()[:30]!r})")
        except Exception as e:
            publish("image-test-failed", err=str(e)[:200])
            log(f"   IMAGE REQUEST FAILED: {str(e)[:200]}")
    return tps


def exercise(cfg, quiet=False):
    """Hit the shapes a real client hits: short and long prompts, streaming, a
    small concurrent batch. In build mode this puts their graphs in the cache;
    in fast_start mode it loads them so users don't hit the one-time stalls."""
    steps = [("short prompt", lambda: completion(cfg, "Hello", 8)),
             ("4k-token prompt", lambda: completion(
                 cfg, "The quick brown fox jumps over the lazy dog. " * 400, 16, timeout=1200)),
             ("streaming", lambda: completion(
                 cfg, "Write a short story about a lighthouse.", 64, stream=True))]
    n = min(cfg["max_num_seqs"], 8)
    errs = []

    def one():
        try:
            completion(cfg, "Count from one to twenty in words.", 48)
        except Exception as e:
            errs.append(str(e)[:200])

    def batch():
        ths = [threading.Thread(target=one) for _ in range(n)]
        [t.start() for t in ths]
        [t.join() for t in ths]
    steps.append((f"{n} parallel requests", batch))
    for name, fn in steps:
        t = time.time()
        try:
            fn()
        except Exception as e:
            errs.append(f"{name}: {str(e)[:200]}")
        if not quiet:
            log(f"   warmed: {name} ({time.time() - t:.0f} s)")
    return errs


# ---------------- maintainer mode: build the env dataset ----------------
BUILD_CONFIGS = [
    {"max_model_len": 262144, "max_num_seqs": 4, "mtp_tokens": 3, "text_only": False},
    {"max_model_len": 131072, "max_num_seqs": 16, "mtp_tokens": 3, "text_only": False},
    {"max_model_len": 262144, "max_num_seqs": 4, "mtp_tokens": 3, "text_only": True},
]
if CFG["build_bundle"]:
    banner(4, "BUILD MODE", "serving each config once to populate the XLA cache")
    log("   TPU-related env:", {k: v for k, v in os.environ.items() if "TPU" in k or "PJRT" in k})
    results = {}
    for c in BUILD_CONFIGS:
        cfg = {**CFG, **c}
        key = (f"{c['max_model_len']}/{c['max_num_seqs']}/mtp{c['mtp_tokens']}"
               + ("/text" if c["text_only"] else "/mm"))
        server = launch_server(cfg)
        secs = wait_healthy(server, cfg, 30)
        errs = exercise(cfg, quiet=True)
        tps = self_test(cfg)
        stop_server(server)
        results[key] = {"startup_secs": secs, "decode_tok_s": tps, "errors": errs}
        publish("build-config-done", config=key, **results[key])
    # probe: how fast is a start with SKIP_JAX_PRECOMPILE=1 now that the cache is warm?
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    cfg = {**CFG, **BUILD_CONFIGS[0]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 5)
    lat = []
    for i in range(3):
        t = time.time()
        try:
            completion(cfg, ["Hello", "Say hi.", "Name a color."][i], 8, timeout=1800)
            lat.append(round(time.time() - t, 1))
        except Exception as e:
            lat.append(str(e)[:100])
    tps = self_test(cfg)
    stop_server(server)
    os.environ.pop("SKIP_JAX_PRECOMPILE")
    publish("probe-skip-precompile", startup_secs=secs, first_request_secs=lat, decode_tok_s=tps)
    # also a warm re-start of the default config (what users will see)
    cfg = {**CFG, **BUILD_CONFIGS[0]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 15)
    tps = self_test(cfg)
    stop_server(server)
    publish("probe-warm-restart", startup_secs=secs, decode_tok_s=tps)
    cfg = {**CFG, **BUILD_CONFIGS[2]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 8)
    tps = self_test(cfg)
    stop_server(server)
    publish("probe-warm-restart-text-only", startup_secs=secs, decode_tok_s=tps)

    out = WORK / "bundle"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    banner(5, "packing the bundle", str(out))
    # (no venv tarball: uv rebuilds the identical env in ~30 s, and Kaggle would
    #  unpack a tar into 100k files anyway — some with '[' in the name, which it rejects)
    sh(["tar", "-cf", str(out / "xla_cache.tar"), "-C", str(SESSION_DIR), "xla_cache"], "tar")
    pkgs = subprocess.run([sys.executable, "-m", "uv", "pip", "list", "--python", PY,
                           "--format=json"], capture_output=True, text=True)
    try:
        pkgs = {d["name"]: d["version"] for d in json.loads(pkgs.stdout)}
    except Exception:
        pkgs = {}
    manifest = {
        "built": time.strftime("%Y-%m-%d"),
        "python": PY_VER,
        "vllm_tpu_version": CFG["vllm_tpu_version"],
        "mtp_patch": "applied at runtime",
        "min_token_bucket": CFG["min_token_bucket"],
        "configs": [[c["max_model_len"], c["max_num_seqs"], c["mtp_tokens"], c["text_only"]]
                    for c in BUILD_CONFIGS],
        "results": results,
        "accelerator": "TPU v5e-8 (Kaggle)",
        "packages": pkgs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    sizes = {p.name: round(p.stat().st_size / 1e9, 2) for p in out.iterdir()}
    publish("bundle-built", sizes_gb=sizes, results=results)
    sys.exit(0)

# ---------------- 4. launch ----------------
expect_min = (10 if n_entries else 20) + (0 if CFG["text_only"] else (10 if n_entries else 15))
if CFG["fast_start"]:
    expect_min = 5 if n_entries else 7
    if not n_entries:
        log("   fast_start without a compile cache: every new request shape will compile "
            "cold; dataset caches are disabled in this fork")
banner(4, "Starting vLLM", f"TP=8, ctx {CFG['max_model_len']}, {CFG['max_num_seqs']} seqs, "
       f"MTP k={CFG['mtp_tokens']}, {'text-only' if CFG['text_only'] else 'multimodal'}")
log(f"   expect ~{expect_min} min; progress lines below, full vLLM log in {RAW_LOG}")
server = launch_server(CFG)

# ---------------- 5. tunnel (in parallel with the server start) ----------------
banner(5, "Public URL")
url = None
tunnel = None
# Recheck immediately before execution as well as during installation.
if not hmac.compare_digest(hashlib.sha256(CLOUDFLARED.read_bytes()).hexdigest(),
                           CLOUDFLARED_SHA256):
    stop_server(server)
    raise RuntimeError("cloudflared changed after verification; refusing to execute")
if CLOUDFLARED.exists():
    tunnel = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{PORT}",
                               "--no-autoupdate", "--protocol", "quic"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    lines = []

    def pump_cf():
        for line in tunnel.stdout:
            lines.append(line.rstrip())
            _raw.write(f"[cloudflared] {line}")
    threading.Thread(target=pump_cf, daemon=True).start()
    deadline = time.time() + 180
    while time.time() < deadline and url is None:
        for ln in lines:
            m = pat.search(ln)
            if m:
                url = m.group(0).rstrip("/")
                break
        time.sleep(1)
if url:
    log(f"   your endpoint will be  {url}/v1")
    log("   (not live yet — it answers 502 until the READY banner below)")
    publish("tunnel-url", endpoint=f"{url}/v1")
else:
    publish("tunnel-failed", note="server still reachable inside the kernel on :8000")

# ---------------- 6. wait, announce, self-test, keep alive ----------------
startup = wait_healthy(server, CFG, expect_min)
publish("serving", startup_secs=startup)
log("")
log("#" * 70)
log(f"#  READY — the server is live ({elapsed()} after start)")
log(f"#  ENDPOINT : {url + '/v1' if url else 'http://127.0.0.1:8000/v1 (tunnel failed)'}")
log(f"#  API key is in private file: {KEY_FILE}")
log("#  CLI users: the launcher displays its locally saved key.")
log(f"#  MODEL    : {CFG['served_model_name']}   (context {CFG['max_model_len']}, "
    f"{CFG['max_num_seqs']} parallel requests)")
log("#" * 70)
log("#  Try it:")
log(f"#    curl {url + '/v1' if url else 'http://127.0.0.1:8000/v1'}/chat/completions \\")
log('#      -H "Authorization: Bearer $INFERENCE_API_KEY" -H "Content-Type: application/json" \\')
log("#      -d '{\"model\": \"" + CFG["served_model_name"] + "\", \"messages\": [{\"role\": \"user\", "
    "\"content\": \"Hello!\"}], \"chat_template_kwargs\": {\"reasoning_effort\": \"low\"}}'")
log(f"#  Serving for up to {CFG['keepalive_min']} min, then this cell exits on its own.")
log("#" * 70)
publish("ready", endpoint=(f"{url}/v1" if url else None),
        model=CFG["served_model_name"], max_model_len=CFG["max_model_len"],
        keepalive_min=CFG["keepalive_min"], startup_secs=startup)

if CFG["fast_start"]:
    banner(6, "Warm-up", "loading the common request shapes; the endpoint is usable meanwhile")
    log("   (fast_start: a request with a new shape waits ~1 min the first time)")
    exercise(CFG)
else:
    banner(6, "Self-test", "one short generation; the endpoint is usable meanwhile")
self_test(CFG)

t_serve = time.time()
while time.time() - t_serve < CFG["keepalive_min"] * 60:
    time.sleep(120)
    if server.poll() is not None:
        publish("stopped", reason="server-exit", rc=server.returncode)
        sys.exit(1)
    up = int((time.time() - t_serve) / 60)
    if up % 10 < 2:
        publish("heartbeat", up_min=up, endpoint=(f"{url}/v1" if url else None))
        log(f"   still serving ({up} min) — {url + '/v1' if url else ''}")
publish("auto-shutdown", served_min=CFG["keepalive_min"])
server.terminate()
sys.exit(0)
