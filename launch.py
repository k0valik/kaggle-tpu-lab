#!/usr/bin/env python3
"""
kaggle-tpu-lab launcher — serve a model on a free Kaggle TPU from your terminal.

    python launch.py serve                          # Qwen3.8-27B: push the kernel and watch it come up
    python launch.py serve --model glm53-flash      # GLM-5.3-Flash
    python launch.py serve --reasoning-effort medium --mtp 3
    python launch.py build-weights --hf-model-id Qwen/Qwen3.8-27B
                                     # mirror HF weights on free CPU (no TPU time burned)
    python launch.py status                # one-shot status + recent events
    python launch.py stop                  # kill the TPU session

Requires the Kaggle CLI, authenticated:  pip install kaggle   (see README).
Only the Python standard library is used here.
"""
import argparse
import base64
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "qwen38-27b" / "kernel" / "serve_qwen38.py"
STATE_FILE = Path.home() / ".kaggle-tpu-lab.json"

WEIGHTS_DATASET = "rahim3/qwen3-8-27b-bf16"
ENV_DATASET = "rahim3/qwen38-tpu-env-v5e8"   # XLA compile cache + cloudflared + manifest
GLM_DATASETS = ["rahim3/glm53-flash-iq3xxs-1", "rahim3/glm53-flash-iq3xxs-2",
                "rahim3/glm53-flash-fp8-1", "rahim3/glm53-flash-fp8-2", "rahim3/glm53-flash-fp8-3", "rahim3/glm53-flash-fp8-4"]

# One entry per model folder: the kernel script, the default kernel name, and (for our own engine) the package to embed.
MODELS = {
    "qwen38-27b": {"kernel": KERNEL_SRC, "slug": "qwen38-tpu-serve", "model_name": "qwen3.8-27b", "minutes": 22},
    "glm53-flash": {"kernel": HERE / "glm53-flash" / "kernel" / "serve_glm53.py", "slug": "glm53-tpu-serve",
                    "model_name": "glm-5.3-flash", "engine": HERE / "glm53-flash" / "engine" / "glm53", "minutes": 25},
}

# Friendly one-liners for each phase the kernel publishes.
PHASE_TEXT = {
    "install":            "Building the Python runtime with uv (~30 s)...",
    "installed":          "Runtime ready.",
    "mtp-patch-applied":  "MTP state-rollback patch applied.",
    "mtp-patch-failed":   "MTP patch did not apply — speculative decoding disabled for safety.",
    "cache-restored":     None,  # rendered below (depends on config coverage)
    "cache-missing":      "No compile cache found — cold compile, add ~10 min.",
    "cache-bypassed":     "Custom checkpoint: prebaked XLA cache ignored — compiling cold.",
    "weights-mounted":    "Weights found mounted (no download needed).",
    "weights-download":   "Downloading weights from Hugging Face (~5 min)...",
    "weights-progress":   "Downloading weights from Hugging Face...",
    "weights-downloaded": "Weights downloaded.",
    "server-launch":      "Starting vLLM — loading the checkpoint weights, then TPU graph compile...",
    "loading":            "Loading the weights onto the chips (~9 min)...",
    "loaded":             None,
    "warmed":             None,
    "tunnel-url":         None,
    "compiling":          None,  # rendered with elapsed time below
    "serving":            "Server is HEALTHY.",
    "benchmark":          None,
    "ready":              None,
    "heartbeat":          None,
    "failed":             None,
    "auto-shutdown":      "Keepalive window ended — kernel shut down cleanly.",
    "stopped":            "Server exited unexpectedly.",
}


def kaggle(*args, capture=True, input=None):
    exe = shutil.which("kaggle")  # prefer the on-PATH console script ...
    cmd = [exe, *args] if exe else [sys.executable, "-m", "kaggle", *args]  # ... else the module
    r = subprocess.run(cmd, capture_output=capture, text=True, input=input)
    return r


def say(msg):
    print(time.strftime("[%H:%M] "), msg, flush=True)


def write_state(state):
    """Atomically replace the state file with owner-only (0600) perms.

    The state holds the API key, so: temp file in the same directory (mkstemp
    is 0600 regardless of umask) -> explicit chmod -> fsync -> atomic replace.
    """
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".ktl-state-", dir=str(STATE_FILE.parent))
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(state, out)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, STATE_FILE)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def api_key_hint(state):
    """What the ready banner shows for the key: the local copy, or where to find it.

    The ntfy `ready` event carries no key (Stage D secret hygiene), so the
    launcher must render it from local state — never from the event payload.
    """
    if state.get("api_key"):
        return state["api_key"]
    if state.get("api_key_secret"):
        return f"(Kaggle Secret: {state['api_key_secret']})"
    return "(see the private Kaggle session output)"


def check_auth():
    r = kaggle("kernels", "list", "-m", "--page-size", "1")
    if r.returncode != 0:
        sys.exit("Kaggle CLI is not working or not authenticated.\n"
                 "Install with `pip install kaggle`, then put your API token in place\n"
                 "(https://www.kaggle.com/settings -> Create New Token).\n\n"
                 f"Error was:\n{(r.stderr or r.stdout).strip()}")


def kaggle_username(cli_arg):
    if cli_arg:
        return cli_arg
    r = kaggle("config", "view")
    m = re.search(r"username[:=]\s*(\S+)", (r.stdout or "") + (r.stderr or ""))
    if m and m.group(1) not in ("None", "-"):
        return m.group(1).strip("'\"")
    sys.exit("Could not detect your Kaggle username — pass it with --user <name>.")


def optional_dataset(name):
    """Normalize a Kaggle dataset arg: '', 'none', 'null' or '-' (any case) means 'not mounted'.

    Returns "" for "no dataset" so callers can filter empties out of
    dataset_sources; an empty value means "HF download inside the kernel",
    never a bogus source.
    """
    if name is None:
        return ""
    s = str(name).strip()
    if s.lower() in ("", "none", "null", "-"):
        return ""
    return s


def engine_b64(pkg_dir):
    """The engine package (its .py files) as a base64 tar.gz, embedded into the kernel script."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in sorted(pkg_dir.glob("*.py")):
            tf.add(f, arcname=f"{pkg_dir.name}/{f.name}")
    return base64.b64encode(buf.getvalue()).decode()


def cmd_serve(args):
    check_auth()
    user = kaggle_username(args.user)
    model = MODELS[args.model]
    slug = args.slug or model["slug"]
    topic = "ktl-" + uuid.uuid4().hex[:20]
    api_key = ("glm-" if args.model == "glm53-flash" else "sk-") + secrets.token_hex(16)

    if args.model == "qwen38-27b":
        weights_dataset = optional_dataset(args.weights_dataset)
        # Stable-key option: --api-key-secret NAME leaves api_key empty and lets the
        # kernel read the Kaggle Secret at runtime; KTL_API_KEY embeds a caller key.
        api_key_secret = (args.api_key_secret or "").strip()
        if api_key_secret:
            api_key = ""
        elif os.environ.get("KTL_API_KEY", "").strip():
            api_key = os.environ["KTL_API_KEY"].strip()
        cfg = {
            "ntfy_topic": topic,
            "api_key": api_key,
            "api_key_secret": api_key_secret,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "mtp_tokens": args.mtp,
            "reasoning_effort_default": args.reasoning_effort,
            "keepalive_min": args.keepalive_min,
            "weights_dataset": weights_dataset,
            "hf_model_id": args.hf_model_id,
            "served_model_name": args.served_model_name,
            "cloudflare_hostname": (args.cloudflare_hostname or "").strip(),
            "cloudflare_token_secret": (args.cloudflare_token_secret or "").strip(),
            "cloudflare_protocol": (args.cloudflare_protocol or "auto").strip(),
        }
        if args.no_tools:
            cfg["tool_call_parser"] = ""
        if args.text_only:
            cfg["text_only"] = True
        if args.verbose:
            cfg["verbose"] = True
        if args.fast_start:
            cfg["fast_start"] = True
        if args.no_async_scheduling:
            cfg["async_scheduling"] = False
        datasets = [d for d in [weights_dataset, ENV_DATASET] if d]
    else:
        cfg = {
            "ntfy_topic": topic,
            "api_key": api_key,
            "max_len": args.max_len,
            "streams": args.streams,
            "reasoning_effort_default": args.reasoning_effort if args.reasoning_effort in ("low", "medium", "high") else "low",
            "keepalive_min": args.keepalive_min,
            "vision": not args.text_only,
        }
        if args.serve_dataset:
            cfg["serve_dataset"] = args.serve_dataset
            datasets = GLM_DATASETS[:2] + [args.serve_dataset]           # experts + the serve dataset; no FP8 mounts
        else:
            datasets = GLM_DATASETS

    src = model["kernel"].read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"{model['kernel']} is missing the __LAUNCHER_CONFIG__ line")
    if model.get("engine"):
        src, n = re.subn(r'^ENGINE_B64 = ""  # __ENGINE__.*$', f'ENGINE_B64 = "{engine_b64(model["engine"])}"',
                         src, count=1, flags=re.M)
        if n != 1:
            sys.exit(f"{model['kernel']} is missing the __ENGINE__ line")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / model["kernel"].name).write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": model["kernel"].name,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "machine_shape": "TpuV5E8",
            "enable_internet": "true",
            "dataset_sources": datasets,
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        say(f"Pushing kernel {user}/{slug} (TPU v5e-8)...")
        r = kaggle("kernels", "push", "-p", str(td), "--accelerator", "TpuV5E8")
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            sys.exit(f"Push failed:\n{out.strip()}")
        if "successfully pushed" not in out.lower():
            say(out.strip())
        for line in out.splitlines():
            if "not valid dataset sources" in line:
                say(f"WARNING: {line.strip()} — the kernel will still run, "
                    "but may need to download weights / compile cold.")

    state = {"kernel": f"{user}/{slug}", "topic": topic, "api_key": api_key,
             "model": args.model, "submitted_at": int(time.time()),
             "keepalive_min": args.keepalive_min}
    if args.model == "qwen38-27b" and (args.api_key_secret or "").strip():
        state["api_key_secret"] = (args.api_key_secret or "").strip()
    write_state(state)
    say(f"Pushed. Kernel page: https://www.kaggle.com/code/{user}/{slug}")
    say("Kaggle takes a few minutes to provision the TPU and attach the "
        f"datasets; the endpoint is usually live ~{model['minutes']} min after the kernel starts.")
    say("Watching progress (Ctrl-C is safe — the server keeps running; "
        "`python launch.py status` re-attaches, `... stop` kills it).")
    watch(f"{user}/{slug}", topic, api_key=api_key_hint(state))


def read_events(topic, since):
    try:
        with urllib.request.urlopen(
                f"https://ntfy.sh/{topic}/json?poll=1&since={since}", timeout=15) as r:
            body = r.read(1024 * 1024).decode()  # read cap: ntfy is progress transport, not a log sink
    except Exception:
        return []
    events = []
    for line in body.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "message":
            continue
        try:
            events.append((e["time"], json.loads(e.get("message", "{}"))))
        except Exception:
            continue
    return events


def render_event(ev, api_key=None):
    """Render one kernel progress event. The ready banner shows the API key from
    local launcher state (api_key arg), never from the ntfy payload — the kernel
    no longer publishes the key (Stage D secret hygiene)."""
    phase = ev.get("phase", "?")
    if phase == "compiling":
        if "what" in ev:
            say(f"Compiled {ev['what']} in {ev.get('secs', 0)} s")
        else:
            say(f"Loading / compiling... {ev.get('elapsed_s', 0) // 60} min elapsed "
                "(typically ~20 min with the env dataset, ~35 min without)")
    elif phase == "loaded":
        say(f"Weights on the chips after {ev.get('minutes', '?')} min (HBM {ev.get('hbm_gb', '?')} GB per chip); warming up...")
    elif phase == "warmed":
        say(f"Warm-up done in {ev.get('minutes', '?')} min; opening the tunnel...")
    elif phase == "cache-restored":
        if ev.get("covers_this_config", True):
            say("XLA compile cache restored for this exact config — fast start.")
        else:
            say("XLA compile cache restored, but not for this config — its graphs "
                "compile cold (add ~10 min).")
    elif phase == "tunnel-url":
        say(f"Endpoint URL reserved: {ev.get('endpoint')}  (not live yet — wait for the banner)")
    elif phase == "tunnel-reconnecting":
        say(f"Tunnel reconnecting over {ev.get('protocol', '?')} in {ev.get('retry_secs', '?')} s "
            f"— {ev.get('endpoint', '')} keeps working after reconnect.")
    elif phase == "serving":
        say(f"Server is HEALTHY after {ev.get('startup_secs', 0) // 60} min.")
    elif phase == "benchmark":
        say(f"Quick benchmark: {ev.get('decode_tok_s', '?')} tok/s single-stream decode "
            f"(sanity: {ev.get('sanity', '')!r})")
    elif phase == "weights-progress":
        say(f"Downloading weights... {ev.get('downloaded_gb', '?')} GB downloaded "
            f"({ev.get('free_gb', '?')} GB free)")
    elif phase == "ready":
        print("\n" + "=" * 66)
        print("  YOUR ENDPOINT IS LIVE")
        print(f"  base URL : {ev['endpoint']}")
        print(f"  API key  : {api_key or ev.get('api_key', '(see the private Kaggle session output)')}")
        print(f"  model    : {ev['model']}   (context: {ev.get('max_model_len', '?')})")
        print("=" * 66)
        base = ev["endpoint"] if ev["endpoint"].endswith("/v1") else ev["endpoint"] + "/v1"
        print(f"""
Try it:
  curl {base}/chat/completions -H "Authorization: Bearer $KEY" \\
    -H "Content-Type: application/json" -d '{{
      "model": "{ev['model']}",
      "messages": [{{"role": "user", "content": "Hello!"}}],
      "chat_template_kwargs": {{"reasoning_effort": "low"}}
    }}'

See the model folder's README for hooking this into Claude Code, Codex CLI, opencode, etc.
""")
        say(f"The kernel keeps serving for up to {ev.get('keepalive_min', '?')} min. "
            "Ctrl-C here does NOT stop it; use `python launch.py stop`.")
    elif phase == "heartbeat":
        say(f"Still serving ({ev.get('up_min', '?')} min up) — {ev.get('endpoint', '')}")
    elif phase == "stopped" and (ev.get("cause") or ev.get("hint") or ev.get("tail")):
        say("Server exited unexpectedly." + (f" Root cause: {ev['cause']}" if ev.get("cause") else ""))
        if ev.get("hint"):
            say(f"Hint: {ev['hint']}")
        if ev.get("tail"):
            print("--- server error ---")
            print(ev["tail"])
    elif phase == "failed":
        say(f"FAILED at step {ev.get('step', '?')}.")
        if ev.get("step") == "no-tpu":
            say("Kaggle started this session without a TPU attached. Nothing in the kernel can fix that: "
                "run `python launch.py stop`, then `serve` again.")
        if ev.get("cause"):
            say(f"Root cause: {ev['cause']}")
        if ev.get("hint"):
            say(f"Hint: {ev['hint']}")
        if ev.get("tail"):
            print("--- last server output ---")
            print(ev["tail"])
        say("Full log: `python launch.py status` after the kernel exits, or the "
            "kernel page on kaggle.com.")
    else:
        text = PHASE_TEXT.get(phase)
        say(text if text else f"{phase} {json.dumps({k: v for k, v in ev.items() if k != 'phase'})}")


def watch(kernel, topic, api_key=None):
    since = int(time.time()) - 600
    last_status = None
    seen_boot = False
    try:
        while True:
            for ts, ev in read_events(topic, since):
                since = max(since, ts)
                seen_boot = True
                render_event(ev, api_key=api_key)
                if ev.get("phase") in ("failed", "auto-shutdown", "stopped"):
                    return
            since = max(since, int(time.time()) - 1) if seen_boot else since
            r = kaggle("kernels", "status", kernel)
            out = (r.stdout or "") + (r.stderr or "")
            m = re.search(r'"KernelWorkerStatus\.(\w+)"', out)
            status = m.group(1) if m else "UNKNOWN"
            if status != last_status:
                if status == "QUEUED":
                    say("Kaggle: queued — waiting for a TPU v5e-8 slot...")
                elif status == "RUNNING" and not seen_boot:
                    say("Kaggle: provisioning the VM and attaching datasets "
                        "(a few minutes)...")
                elif status == "UNKNOWN":
                    if "permission 'kernels.get' was denied" in out.lower() or "cannot access kernel" in out.lower():
                        say("Kaggle API cannot verify this private kernel yet; "
                            "waiting for status/events instead of assuming it is queued.")
                    else:
                        say("Kaggle status unavailable; retrying without assuming a queue state.")
                elif status in ("ERROR", "CANCELACKNOWLEDGED", "COMPLETE"):
                    say(f"Kernel finished with status {status}.")
                    return
                last_status = status
            time.sleep(30)
    except KeyboardInterrupt:
        say("Detached. The kernel keeps running — `python launch.py status` to "
            "re-attach, `python launch.py stop` to kill it.")


# Template for the free-CPU weights-mirror kernel pushed by `build-weights`
# (Stage B2). __REPO_ID__ is replaced with the JSON-quoted --hf-model-id at
# push time. Everything else is runtime logic: the HF token is NEVER baked
# into the pushed source — gated repos authenticate inside the kernel via the
# HF_TOKEN env var or the Kaggle Secret labelled HF_TOKEN.
BUILD_WEIGHTS_TEMPLATE = '''import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ID = __REPO_ID__

ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.jinja", "*.txt",
                  "tokenizer*", "vocab*", "merges*"]
OUT_DIR = Path("/kaggle/working/weights")


def resolve_token():
    """Runtime-only HF auth: env HF_TOKEN first, else the Kaggle Secret
    labelled HF_TOKEN (attach it under the kernel's Secrets add-on for
    gated repos). Returns None for public repos."""
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        tok = (UserSecretsClient().get_secret("HF_TOKEN") or "").strip()
        return tok or None
    except Exception:
        return None


try:
    from huggingface_hub import snapshot_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "huggingface_hub"])
    from huggingface_hub import snapshot_download

print("Downloading %s to %s ..." % (REPO_ID, OUT_DIR), flush=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.time()
snapshot_download(repo_id=REPO_ID, local_dir=str(OUT_DIR),
                  allow_patterns=ALLOW_PATTERNS, token=resolve_token())
secs = int(time.time() - t0)
files = [p for p in OUT_DIR.rglob("*") if p.is_file()]
total_gb = sum(p.stat().st_size for p in files) / 1e9
print("Download complete in %ds: %d files, %.2f GB in %s"
      % (secs, len(files), total_gb, OUT_DIR), flush=True)
print("")
print("Next steps (no TPU time burned):")
print("  1. On this kernel's page, wait for status COMPLETE.")
print("  2. Output tab -> New Dataset -> save /kaggle/working/weights.")
print("  3. Attach it to the TPU run (kernel Add Input in the UI, or")
print("     python launch.py serve --weights-dataset <user>/<dataset-slug>).")
'''


def cmd_build_weights(args):
    """Push a free CPU kernel that mirrors a Hugging Face checkpoint to
    /kaggle/working/weights (Stage B2). No TPU quota is burned on the ~55 GB
    download; the owner creates a Kaggle dataset from the kernel output in
    the UI and attaches it to the TPU run via --weights-dataset."""
    check_auth()
    user = kaggle_username(args.user)
    repo = (args.hf_model_id or "").strip()
    if not repo:
        sys.exit("--hf-model-id must not be empty.")
    slug = args.slug
    kernel = f"{user}/{slug}"
    say(f"Preparing CPU weights-mirror kernel for {repo} ...")
    script = BUILD_WEIGHTS_TEMPLATE.replace("__REPO_ID__", json.dumps(repo))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "download_weights.py").write_text(script)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": kernel, "title": slug, "code_file": "download_weights.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            # CPU-only: no "machine_shape", no --accelerator push flag (the
            # TPU serve path sends machine_shape TpuV5E8 + --accelerator
            # TpuV5E8). No dataset_sources: this kernel downloads from HF.
            "enable_gpu": "false", "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        say(f"Pushing free CPU kernel {kernel} (no --accelerator: CPU default)...")
        r = kaggle("kernels", "push", "-p", str(td))
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            sys.exit(f"Push failed:\n{out.strip()}")
        if "successfully pushed" not in out.lower():
            say(out.strip())
    say(f"Pushed. Kernel page: https://www.kaggle.com/code/{user}/{slug}")
    say("Polling kernel status (Ctrl-C is safe — the kernel keeps running)...")
    # Finite job: poll `kernels status` only, no ntfy topic/watch() flow.
    deadline = time.time() + 3 * 3600
    last = None
    try:
        while True:
            r = kaggle("kernels", "status", kernel)
            out = (r.stdout or "") + (r.stderr or "")
            m = re.search(r'"KernelWorkerStatus\.(\w+)"', out)
            status = m.group(1) if m else "UNKNOWN"
            if status != last:
                say(f"Kernel status: {status}")
                last = status
            if status == "COMPLETE":
                break
            if status == "ERROR" or status.startswith("CANCEL"):
                sys.exit(f"Kernel finished with status {status}:\n{out.strip()}\n"
                         f"See https://www.kaggle.com/code/{user}/{slug}")
            if time.time() > deadline:
                sys.exit(f"Timed out waiting for {kernel} (last status: {last}).\n"
                         f"Check https://www.kaggle.com/code/{user}/{slug}")
            time.sleep(60)
    except KeyboardInterrupt:
        say("Detached. The kernel keeps running — re-run "
            "`python launch.py build-weights` is NOT needed; check the kernel page.")
        return
    print("")
    print("Weights mirrored. Create the dataset from the kernel output:")
    print(f"  1. Open https://www.kaggle.com/code/{user}/{slug} (status COMPLETE).")
    print("  2. Output tab -> New Dataset -> save /kaggle/working/weights.")
    print("  3. Attach it to the TPU run (kernel Add Input in the UI, or")
    print("     python launch.py serve --weights-dataset <user>/<dataset-slug>).")


def cmd_build_env(args):
    """Maintainer flow. When the kernel finishes:
        kaggle kernels output <user>/<slug> -p bundle_out
        then create/version the dataset from bundle_out/bundle (see README)."""
    check_auth()
    user = kaggle_username(args.user)
    topic = "ktl-" + uuid.uuid4().hex[:20]
    # B1 note: build-env intentionally passes only --weights-dataset. It builds the
    # env/XLA-cache bundle against the default BF16 weights; hf_model_id /
    # served_model_name are serve-time weights-source options with no corresponding
    # build-env flags, so there is nothing trivially consistent to pass through here.
    cfg = {"build_bundle": True, "ntfy_topic": topic, "weights_dataset": args.weights_dataset}
    src = KERNEL_SRC.read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit("qwen38-27b/kernel/serve_qwen38.py is missing the __LAUNCHER_CONFIG__ line")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "build_env.py").write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{args.slug}", "title": args.slug, "code_file": "build_env.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "false", "machine_shape": "TpuV5E8", "enable_internet": "true",
            "dataset_sources": [args.weights_dataset],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        r = kaggle("kernels", "push", "-p", str(td), "--accelerator", "TpuV5E8")
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            sys.exit(f"Push failed:\n{out.strip()}")
        if "successfully pushed" not in out.lower():
            say(out.strip())
    write_state({"kernel": f"{user}/{args.slug}", "topic": topic,
                 "api_key": "", "submitted_at": int(time.time())})
    say(f"Pushed {user}/{args.slug}. It serves each config once (~1.5 h total) and "
        "leaves xla_cache.tar / cloudflared / manifest.json in its output.")
    watch(f"{user}/{args.slug}", topic)


def load_state():
    if not STATE_FILE.exists():
        sys.exit("No launch state found — run `python launch.py serve` first.")
    return json.loads(STATE_FILE.read_text())


def cmd_status(args):
    st = load_state()
    say(f"Kernel: {st['kernel']}")
    r = kaggle("kernels", "status", st["kernel"])
    say(((r.stdout or "") + (r.stderr or "")).strip())
    events = read_events(st["topic"], int(time.time()) - 24 * 3600)
    for _, ev in events[-8:]:
        render_event(ev, api_key=api_key_hint(st))
    if any(ev.get("phase") == "ready" for _, ev in events):
        say(f"API key: {api_key_hint(st)}")
    if args.follow:
        watch(st["kernel"], st["topic"], api_key=api_key_hint(st))


def cmd_stop(args):
    st = load_state()
    say(f"Deleting kernel {st['kernel']} (terminates the TPU session)...")
    p = kaggle("kernels", "delete", st["kernel"], input="yes\n")
    out = (p.stdout + p.stderr).strip()
    say(out or "done")
    if p.returncode != 0:
        sys.exit(f"Stop failed:\n{out or f'kaggle kernels delete exited {p.returncode}'}")
    STATE_FILE.unlink(missing_ok=True)
    say("Local session state cleared.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="push the serving kernel and watch it come up")
    s.add_argument("--model", default="qwen38-27b", choices=sorted(MODELS), help="which recipe (model folder) to serve")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default=None, help="kernel name (default: the model's)")
    s.add_argument("--max-len", type=int, default=262144, help="glm53-flash: context capacity (a multiple of 32)")
    s.add_argument("--streams", type=int, default=4, help="glm53-flash: requests decoded together")
    s.add_argument("--serve-dataset", default="rahim3/glm53-flash-serve",
                   help="glm53-flash: dataset with base/ + jax_cache/ (replaces the FP8 mounts, shorter warm-up); '' = FP8 mounts, cold")
    s.add_argument("--max-model-len", type=int, default=262144,
                   help="context length (default: native 262k; use 131072 with "
                        "--max-num-seqs 16 for max multi-stream throughput)")
    s.add_argument("--max-num-seqs", type=int, default=4)
    s.add_argument("--mtp", type=int, default=3,
                   help="MTP speculative tokens (0 disables). +34%% decode in our A/B test; made "
                        "lossless by the bundled GDN state-rollback patch "
                        "(verified 12/12 greedy exact-match)")
    s.add_argument("--reasoning-effort", default="xhigh",
                   choices=["xhigh", "high", "medium", "low"],
                   help="server-side default; clients can still override per request "
                        "(qwen38-27b: xhigh | medium | low; glm53-flash: low | medium | high, default low)")
    s.add_argument("--keepalive-min", type=int, default=480,
                   help="auto-shutdown after this many minutes of serving")
    s.add_argument("--weights-dataset", default=WEIGHTS_DATASET,
                     help="qwen38-27b: Kaggle dataset with the weights (e.g. your own FP8 quant). "
                          "'', 'none', 'null' or '-' (case-insensitive) = no dataset mount; "
                          "the kernel downloads from Hugging Face instead (see --hf-model-id). "
                          f"(default: {WEIGHTS_DATASET})")
    s.add_argument("--hf-model-id", default="Qwen/Qwen3.8-27B",
                     help="qwen38-27b: Hugging Face repo downloaded when no weights dataset "
                          "is mounted (i.e. --weights-dataset none)")
    s.add_argument("--served-model-name", default="qwen3.8-27b",
                     help="qwen38-27b: model name advertised by the OpenAI-compatible API")
    s.add_argument("--cloudflare-hostname", default="",
                     help="qwen38-27b: fixed hostname of a Cloudflare named tunnel "
                          "(e.g. llm.example.com). Empty (default) = random quick tunnel. "
                          "The tunnel token is read inside Kaggle from --cloudflare-token-secret.")
    s.add_argument("--cloudflare-token-secret", default="CF_TUNNEL_TOKEN",
                     help="qwen38-27b: Kaggle Secret label holding the named-tunnel token "
                          "(only used with --cloudflare-hostname; never passed on argv)")
    s.add_argument("--cloudflare-protocol", default="auto", choices=["auto", "http2", "quic"],
                     help="qwen38-27b: first transport for a named tunnel; the kernel "
                          "alternates auto<->http2 on disconnects")
    s.add_argument("--api-key-secret", default="",
                     help="qwen38-27b: Kaggle Secret label holding a stable endpoint API key. "
                          "Empty (default) = the launcher generates a fresh key (or uses "
                          "KTL_API_KEY from the environment).")
    s.add_argument("--no-tools", action="store_true",
                   help="disable tool-calling support")
    s.add_argument("--text-only", action="store_true",
                   help="skip the vision tower (Qwen: ~8 min faster start; GLM: ~1 min); image inputs "
                        "then error out")
    s.add_argument("--verbose", action="store_true",
                   help="show every vLLM log line in the kernel log")
    s.add_argument("--no-async-scheduling", action="store_true",
                   help="qwen38-27b: pass --no-async-scheduling to vLLM. Needed when clients use JSON mode / "
                        "structured outputs with MTP on (vllm-tpu 0.29.0 otherwise exits with AttributeError: "
                        "__delitem__); costs some throughput")
    s.add_argument("--fast-start", action="store_true",
                   help="skip TPU graph precompile: endpoint live in ~4 min (with the env "
                        "dataset), common request shapes are warmed right after; an "
                        "unusual request shape stalls ~1 min the first time")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("build-weights", help="push a free CPU kernel that mirrors "
                       "a Hugging Face checkpoint for dataset creation (no TPU time)")
    s.add_argument("--hf-model-id", required=True,
                   help="Hugging Face repo to mirror (e.g. Qwen/Qwen3.8-27B). "
                        "Required, no default — no checkpoint is ever hardcoded. "
                        "Gated repos authenticate at runtime via the HF_TOKEN "
                        "Kaggle Secret (never embedded in the pushed source).")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="qwen38-weights-mirror",
                   help="kernel name (default: %(default)s)")
    s.set_defaults(fn=cmd_build_weights)

    s = sub.add_parser("build-env", help="(maintainers) push a kernel that builds the "
                       "env dataset: venv + XLA cache + cloudflared")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="qwen38-env-bundle")
    s.add_argument("--weights-dataset", default=WEIGHTS_DATASET)
    s.set_defaults(fn=cmd_build_env)

    s = sub.add_parser("status", help="show current kernel status + recent events")
    s.add_argument("--follow", "-f", action="store_true", help="keep watching")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("stop", help="terminate the TPU session")
    s.set_defaults(fn=cmd_stop)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
