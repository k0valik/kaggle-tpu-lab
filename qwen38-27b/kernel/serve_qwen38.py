"""
Serve Qwen3.8-27B (bf16) on a Kaggle TPU v5e-8 with vLLM.

This script is pushed to Kaggle as a script kernel by ../launch.py, which fills
in the CFG line below. It also runs standalone with defaults (e.g. pasted into
a Kaggle notebook/script in the UI) — then it just prints instead of using ntfy.

Steps (each one is announced in the log):
  1/6  runtime  — venv with vllm-tpu (pinned, CPU torch) built by uv in ~30 s,
                  resolution pinned to the env dataset's build date; + the MTP fix
  2/6  cache    — restore the pre-built XLA compile cache from the env dataset
  3/6  weights  — find the mounted weights dataset (or download from HF to /tmp)
  4/6  server   — start vLLM (TP=8, text-only, MTP speculative decoding)
  5/6  tunnel   — open a public cloudflared URL (printed before the server is
                  live so you can prepare your client)
  6/6  ready    — READY banner + self-test, then keep serving until
                  keepalive_min elapses

With both datasets attached the endpoint is live in ~22 minutes (~12 with
text_only, ~6 with fast_start). Without the env dataset the compile is cold (+15 min).
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

CFG = None  # __LAUNCHER_CONFIG__  (launch.py replaces this line)

DEFAULTS = {
    "vllm_tpu_version": "0.29.0",
    "weights_dataset": "rahim3/qwen3-8-27b-bf16",     # HF mirror of Qwen/Qwen3.8-27B
    "env_dataset": "rahim3/qwen38-tpu-env-v5e8",       # XLA cache + cloudflared + manifest
    "hf_model_id": "Qwen/Qwen3.8-27B",                # fallback download source
    "max_model_len": 262144,       # native context; drop to 131072 + max_num_seqs 16 for throughput
    "max_num_seqs": 4,
    "mtp_tokens": 3,               # MTP spec decoding (+34% in our A/B test). Stock vllm-tpu
                                   # 0.29.0 corrupts outputs with it (missing GDN state
                                   # rollback); we apply patches/mtp-rollback-v0290.diff
                                   # (a port of upstream PR #3178) before serving —
                                   # verified lossless, 12/12 greedy exact-match.
    "async_scheduling": None,      # None = vLLM's default (on). false passes --no-async-scheduling: needed by
                                   # clients that use JSON mode / structured outputs while MTP is on (vllm-tpu
                                   # 0.29.0 hands the scheduler device arrays on that path -> "AttributeError:
                                   # __delitem__" and the server exits); costs some throughput
    "reasoning_effort_default": "xhigh",   # server-side default: xhigh | medium | low
    "tool_call_parser": "qwen3_coder",  # matches Qwen3.8's XML tool format; "" disables
    "text_only": False,            # True: skip the vision tower + its TPU graphs (saves ~8 min,
                                   # image inputs then error out)
    "min_token_bucket": 64,        # smallest padded batch (tokens); 16 = more graphs to compile
    "precompile_workers": 4,       # parallel XLA compile threads (1 = sequential)
    "fast_start": False,           # True: skip precompile -> READY in ~4 min (needs the env
                                   # dataset's cache); the script then warms the common
                                   # request shapes itself; rare shapes stall once (~1 min)
    "keepalive_min": 480,          # auto-shutdown guard (Kaggle TPU caps at 9h anyway)
    "api_key": "",                 # generated if empty
    "ntfy_topic": "",              # optional: publish progress to ntfy.sh/<topic>
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
VENV = "/tmp/venv"
PY = f"{VENV}/bin/python"
XLA_CACHE = "/tmp/xla_cache"
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
RAW_LOG = WORK / "vllm.log"          # every line vLLM/pip print, for debugging
CLOUDFLARED = Path("/tmp/cloudflared")
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


def log(*parts):
    line = time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts)
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
    """Progress event: always logged; also pushed to ntfy if a topic is set."""
    log(f"PHASE {phase}", json.dumps(extra) if extra else "")
    if not CFG["ntfy_topic"]:
        return
    try:
        body = {"topic": CFG["ntfy_topic"], "title": f"kaggle-tpu-lab {phase}",
                "message": json.dumps({"phase": phase, **extra})}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log(f"(ntfy publish failed: {e})")


def sh(cmd, tag, show=None, env=None):
    """Run a command; stream its output to vllm.log (and to the console when
    show/verbose). Returns the exit code."""
    show = CFG["verbose"] if show is None else show
    tail = collections.deque(maxlen=40)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env)
    for line in p.stdout:
        line = line.rstrip()
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
    if CLOUDFLARED.exists():
        return
    try:
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
    except Exception as e:
        log(f"(cloudflared download failed: {e})")


# gzip+base64 of patches/mtp-rollback-v0290.diff; regenerated by tools/embed_patch.py
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
    _raw.write(p.stdout + p.stderr)
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


def tpu_check():
    """Kaggle sometimes starts a "TPU" session with no TPU attached (a CPU-only container; most often on new or
    not-yet-verified accounts). jax then sees one device and vLLM dies minutes later with "Insufficient devices for
    2D mesh: found 1, expected 8" or "No jellyfish device found". Look before installing anything (~20 s)."""
    code = ("import jax\n"
            "try:\n"
            "    d = jax.devices()\n"
            "    print('TPU_CHECK', len(d), d[0].platform, getattr(d[0], 'device_kind', ''))\n"
            "except Exception as e:\n"
            "    print('TPU_CHECK 0 none', str(e).replace(chr(10), ' ')[:200])\n")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"TPU_CHECK (\d+) (\S+)(.*)", out)
    except Exception as e:  # noqa: BLE001
        log(f"   (TPU check skipped: {e})")
        return
    if m is None:
        log("   (TPU check skipped: the image's jax did not answer; details in vllm.log)")
        _raw.write("[tpu-check] " + out.replace("\n", "\n[tpu-check] ") + "\n")
        return
    n, platform, rest = int(m.group(1)), m.group(2), m.group(3).strip()
    if n == 8 and platform == "tpu":
        log(f"   TPU check OK: 8 chips ({rest})")
        return
    if n == 0 and not re.search(r"jellyfish|TPU initialization failed|initialize backend 'tpu'|No TPU|vfio", rest, re.I):
        log(f"   (TPU check inconclusive, continuing: {rest[:160]})")   # only a clear no-TPU signal stops the run
        return
    msg = (f"this session has no working TPU: jax sees {n} {platform} device(s) {rest}. Kaggle sometimes starts a "
           "TPU session without one (most often on new or not-yet-verified accounts); nothing in this notebook can "
           "fix that. Stop the session and start it again. If it repeats, run `import jax; print(jax.device_count())` "
           "in a fresh cell first: it must print 8 before this script is worth running.")
    log("   " + msg)
    publish("failed", step="no-tpu", tail=msg)
    sys.exit(1)


def server_death_report(max_lines=60):
    """What killed vLLM, from vllm.log: the root-cause exception line, the first error block and a hint for the
    causes we have seen. The console tail alone scrolls the cause away. Returns (cause, block, hint)."""
    try:
        lines = [l for l in RAW_LOG.read_text(errors="replace").splitlines() if l.startswith("[vllm]")]
    except Exception:  # noqa: BLE001
        return "", "", ""
    strip = re.compile(r"^\[vllm\] (?:\((?:EngineCore|APIServer|Worker)[^)]*\) )?(?:ERROR|CRITICAL) [\d-]+ [\d:]+ \[[^\]]+\] ?")
    start_pat = re.compile(r"EngineCore (?:failed|encountered|hit)|Traceback \(most recent call last\)|RESOURCE_EXHAUSTED|INVALID_ARGUMENT|NOT_FOUND")
    start = next((i for i, l in enumerate(lines) if start_pat.search(l)), None)
    block = [strip.sub("", l) for l in lines[start:start + max_lines]] if start is not None else []
    cause = ""
    for l in block:                                   # the first traceback's own exception line is the root cause
        t = l.strip()
        if re.match(r"^[\w.]*(?:Error|Exception)\b.*:", t) or re.match(r"^(?:INVALID_ARGUMENT|NOT_FOUND|RESOURCE_EXHAUSTED)", t):
            cause = t
            break
    joined = "\n".join(block)
    hint = ""
    if re.search(r"found 1, expected 8|jellyfish|unexpected worker hostname|TPU initialization failed", joined):
        hint = ("this session has no working TPU (Kaggle sometimes starts one without, most often on new or "
                "not-yet-verified accounts): stop the session and start it again")
    elif "__delitem__" in joined:
        hint = ("a client sent a JSON-mode / structured-output request while MTP and async scheduling are on, which "
                "vllm-tpu 0.29.0 cannot handle: set \"async_scheduling\": false (or \"mtp_tokens\": 0) and run again")
    elif "RESOURCE_EXHAUSTED" in joined or "out of memory" in joined.lower():
        hint = "the TPU ran out of HBM: lower max_model_len or max_num_seqs"
    return cause, joined, hint


def server_died(server, phase, **extra):
    """Log why the vLLM server exited (root cause first), publish it, and stop the kernel."""
    cause, block, hint = server_death_report()
    tail = "\n".join(list(server.tail)[-40:]) if getattr(server, "tail", None) else ""
    log(f"server exited rc={server.returncode}" + (f" — root cause: {cause}" if cause else ""))
    if hint:
        log(f"   -> {hint}")
    if block:
        log("--- first error block from vllm.log ---\n" + block)
    if tail and not block:
        log("--- last output ---\n" + tail)
    log(f"full log: {RAW_LOG}")
    head = (f"root cause: {cause}\n" if cause else "") + (f"{hint}\n" if hint else "")
    publish(phase, rc=server.returncode, cause=cause, hint=hint,
            tail=(head + "\n" + (block or tail)[-2200:]).strip(), **extra)
    sys.exit(1)


# ---------------- 1. runtime ----------------
banner(1, "Python runtime", f"vllm-tpu {CFG['vllm_tpu_version']}")
tpu_check()
threading.Thread(target=fetch_cloudflared, daemon=True).start()
bundle_root = find_input(CFG["env_dataset"].split("/")[-1], "qwen38-tpu-env*")
bundle, manifest = None, {}
if bundle_root:
    # Kaggle may keep the files at the top level or under the kernel's output folder
    hits = glob.glob(f"{bundle_root}/manifest.json") + glob.glob(f"{bundle_root}/*/manifest.json")
    if hits:
        bundle = os.path.dirname(hits[0])
        manifest = json.loads(Path(hits[0]).read_text())
    else:
        bundle = bundle_root
if bundle and Path(bundle, "cloudflared").exists() and not CLOUDFLARED.exists():
    shutil.copy(Path(bundle, "cloudflared"), CLOUDFLARED)
    CLOUDFLARED.chmod(0o755)
if manifest and (manifest.get("python") != PY_VER
                 or manifest.get("vllm_tpu_version") != CFG["vllm_tpu_version"]):
    log(f"   env dataset was built for python {manifest.get('python')} / vllm-tpu "
        f"{manifest.get('vllm_tpu_version')}; this session has python {PY_VER} and wants "
        f"vllm-tpu {CFG['vllm_tpu_version']} -> its compile cache will not match")
    manifest = {}
if not bundle:
    log(f"   env dataset not attached (expected {CFG['env_dataset']}) -> cold compile later")

t = time.time()
publish("install", vllm_tpu=CFG["vllm_tpu_version"])
log("   installer output goes to", RAW_LOG)
runtime = install_runtime(manifest.get("built"))
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
banner(2, "XLA compile cache")
t = time.time()
cache_tar = (Path(bundle, "xla_cache.tar") if bundle and Path(bundle, "xla_cache.tar").exists()
             else find_input("*/xla_cache*.tar.gz", "xla_cache*.tar.gz"))
cache_dir = find_input("*/*/xla_cache", "*/xla_cache", "xla_cache")
if cache_tar:
    flags = "-xf" if str(cache_tar).endswith(".tar") else "-xzf"
    sh(["tar", flags, str(cache_tar), "-C", "/tmp"], "tar")
elif cache_dir:
    sh(["cp", "-r", cache_dir, "/tmp/"], "cp")
    sh(["chmod", "-R", "u+w", XLA_CACHE], "chmod")
n_entries = len(glob.glob(XLA_CACHE + "/*"))
cache_configs = manifest.get("configs", [])
this_config = [CFG["max_model_len"], CFG["max_num_seqs"], CFG["mtp_tokens"], CFG["text_only"]]
if n_entries:
    covered = (not cache_configs) or (this_config in cache_configs)
    publish("cache-restored", entries=n_entries, secs=int(time.time() - t),
            covers_this_config=covered)
    if not covered:
        log(f"   note: the cache was built for [ctx, seqs, mtp, text_only] in {cache_configs}; "
            f"this run uses {this_config} -> its graphs compile cold (add ~10-15 min)")
    else:
        log("   compiled TPU graphs for this exact config are cached -> fast start")
else:
    publish("cache-missing", note="cold compile: expect ~10 extra minutes")


def complete_bf16_repo(path):
    """Validate a mounted bf16 weights mirror. Returns (ok, detail).

    Strictly reject an invalid/missing safetensors index or missing shards.
    The Hugging Face size cross-check is best-effort when the API is reachable.
    """
    idx = os.path.join(path, "model.safetensors.index.json")
    try:
        with open(idx) as f:
            weight_map = json.load(f)["weight_map"]
    except Exception as e:
        return False, f"index missing/invalid ({e})"
    if not isinstance(weight_map, dict) or not weight_map:
        return False, "index has empty weight_map"
    shards = sorted(set(weight_map.values()))
    missing = [v for v in shards if not os.path.isfile(os.path.join(path, v))]
    if missing:
        return False, f"{len(missing)} shard(s) missing (e.g. {missing[0]})"
    total_local = sum(os.path.getsize(os.path.join(path, v)) for v in shards)
    if not any("mtp" in n.lower() for n in weight_map):
        log("   warn: no MTP tensor names in index — serving without speculative decoding")
    try:
        url = f"https://huggingface.co/api/models/{CFG['hf_model_id']}?blobs=true"
        with urllib.request.urlopen(url, timeout=15) as r:
            siblings = json.loads(r.read().decode())["siblings"]
        api_sizes = {s["rfilename"].split("/")[-1]: s.get("size")
                     for s in siblings if isinstance(s, dict)}
        total_api = sum(api_sizes[b] for b in (os.path.basename(v) for v in shards)
                        if isinstance(api_sizes.get(b), int))
        if total_api and abs(total_local - total_api) > 1024:
            return False, f"size mismatch: local {total_local} vs HF {total_api}"
    except Exception as e:
        log(f"   note: HF size check skipped ({e})")
    return True, f"{len(shards)} shards, {total_local / 1e9:.1f} GB verified"


# ---------------- 3. weights ----------------
banner(3, "Model weights", "55 GB bf16 safetensors")
weights_slug = CFG["weights_dataset"].split("/")[-1]
model_path = find_input(weights_slug)
detail = "no local mirror"
if model_path and os.path.exists(os.path.join(model_path, "config.json")):
    ok, detail = complete_bf16_repo(model_path)
    if ok:
        publish("weights-mounted", path=model_path, detail=detail)
    else:
        log(f"   mirror incomplete ({detail}) -> downloading from HF")
        model_path = None
if not model_path:
    publish("weights-download", model=CFG["hf_model_id"],
            note="attach the weights dataset to skip this (~5 min parallel download)",
            reason=detail)
    t = time.time()
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(CFG["hf_model_id"], allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "*.jinja", "tokenizer*", "vocab*", "merges*"])
    try:
        dl = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        total = sum(os.path.getsize(os.path.join(model_path, f)) for f in dl)
        log(f"   downloaded {len(dl)} shards, {total / 1e9:.1f} GB")
    except Exception as e:
        log(f"   (download inventory skipped: {e})")
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
            "--port", str(PORT),
            "--api-key", cfg["api_key"],
            "--served-model-name", cfg["served_model_name"],
            "--reasoning-parser", "qwen3"]
    if cfg.get("async_scheduling") is not None:
        args.append("--async-scheduling" if cfg["async_scheduling"] else "--no-async-scheduling")
    if cfg["text_only"]:
        # Qwen3.8 is a vision-language checkpoint; we only serve text. This skips
        # the vision tower and roughly halves the number of TPU graphs to compile.
        args += ["--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if cfg["mtp_tokens"] > 0:
        args += ["--speculative-config",
                 json.dumps({"method": "mtp", "num_speculative_tokens": cfg["mtp_tokens"]})]
    if cfg["tool_call_parser"]:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", cfg["tool_call_parser"]]
    # 0.29.0: upstream #3422 fixed hybrid-GDN prefix caching; the platform
    # force-disables it under spec-decode, so this is safe with MTP on
    # (guarded off while MTP active) — needs live test to confirm the gain.
    args.append("--enable-prefix-caching")
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
            line = line.rstrip()
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
            server_died(server, "failed", step="server")
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
    sh(["tar", "-cf", str(out / "xla_cache.tar"), "-C", "/tmp", "xla_cache"], "tar")
    fetch_cloudflared()
    if CLOUDFLARED.exists():
        shutil.copy(CLOUDFLARED, out / "cloudflared")
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
            "cold (~1 min each) — attach the env dataset for this mode to make sense")
banner(4, "Starting vLLM", f"TP=8, ctx {CFG['max_model_len']}, {CFG['max_num_seqs']} seqs, "
       f"MTP k={CFG['mtp_tokens']}, {'text-only' if CFG['text_only'] else 'multimodal'}")
log(f"   expect ~{expect_min} min; progress lines below, full vLLM log in {RAW_LOG}")
server = launch_server(CFG)

# ---------------- 5. tunnel (in parallel with the server start) ----------------
banner(5, "Public URL")
url = None
tunnel = None
for _ in range(60):  # cloudflared download runs in the background from step 1
    if CLOUDFLARED.exists():
        break
    time.sleep(2)
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
log(f"#  API KEY  : {CFG['api_key']}")
log(f"#  MODEL    : {CFG['served_model_name']}   (context {CFG['max_model_len']}, "
    f"{CFG['max_num_seqs']} parallel requests)")
log("#" * 70)
log("#  Try it:")
log(f"#    curl {url + '/v1' if url else 'http://127.0.0.1:8000/v1'}/chat/completions \\")
log(f"#      -H 'Authorization: Bearer {CFG['api_key']}' -H 'Content-Type: application/json' \\")
log("#      -d '{\"model\": \"" + CFG["served_model_name"] + "\", \"messages\": [{\"role\": \"user\", "
    "\"content\": \"Hello!\"}], \"chat_template_kwargs\": {\"reasoning_effort\": \"low\"}}'")
log(f"#  Serving for up to {CFG['keepalive_min']} min, then this cell exits on its own.")
log("#" * 70)
publish("ready", endpoint=(f"{url}/v1" if url else None), api_key=CFG["api_key"],
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
        server_died(server, "stopped", reason="server-exit")
    up = int((time.time() - t_serve) / 60)
    if up % 10 < 2:
        publish("heartbeat", up_min=up, endpoint=(f"{url}/v1" if url else None))
        log(f"   still serving ({up} min) — {url + '/v1' if url else ''}")
publish("auto-shutdown", served_min=CFG["keepalive_min"])
server.terminate()
sys.exit(0)
