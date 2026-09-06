#!/usr/bin/env python3
"""
Launch the OrcaRouter Qwen3.8-27B Uncensored FP8 server on a free Kaggle TPU v5e-8.

The model is gated on Hugging Face. Accept its access conditions first, then add
a Kaggle Secret named HF_TOKEN. The launcher never embeds or prints that token.

Examples:
    python launch_fp8.py serve
    python launch_fp8.py serve --text-only --max-model-len 131072
    python launch_fp8.py status -f
    python launch_fp8.py stop
"""

import argparse
import json
import re
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRAPPER_SRC = HERE / "kernel" / "serve_qwen38_fp8.py"
BASE_KERNEL_SRC = HERE / "kernel" / "serve_qwen38.py"
STATE_FILE = Path.home() / ".kaggle-tpu-lab-fp8.json"

MODEL_ID = "orcarouter/Qwen3.8-27B-Uncensored-FP8"
SERVED_MODEL_NAME = "qwen3.8-27b-uncensored-fp8"


def kaggle(*args):
    return subprocess.run(
        [sys.executable, "-m", "kaggle", *args],
        capture_output=True,
        text=True,
    )


def say(message):
    print(time.strftime("[%H:%M] "), message, flush=True)


def check_auth():
    r = kaggle("kernels", "list", "-m", "--page-size", "1")
    if r.returncode:
        raise SystemExit(
            "Kaggle CLI is unavailable or unauthenticated. Install `kaggle`, "
            "configure your Kaggle API token, then retry.\n\n"
            + (r.stderr or r.stdout or "").strip()
        )


def kaggle_username(cli_arg):
    if cli_arg:
        return cli_arg
    r = kaggle("config", "view")
    m = re.search(r"username[:=]\s*(\S+)", (r.stdout or "") + (r.stderr or ""))
    if m and m.group(1) not in ("None", "-"):
        return m.group(1).strip("'\"")
    raise SystemExit("Could not detect your Kaggle username; pass --user <name>.")


def inject_cfg(source, cfg):
    source, count = re.subn(
        r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
        f"CFG = {cfg!r}",
        source,
        count=1,
        flags=re.M,
    )
    if count != 1:
        raise SystemExit("kernel/serve_qwen38_fp8.py is missing the launcher config marker.")
    return source


def dataset_sources(args):
    return [x for x in (args.weights_dataset, args.env_dataset) if x]


def cmd_serve(args):
    check_auth()
    user = kaggle_username(args.user)
    topic = "ktl-fp8-" + uuid.uuid4().hex[:18]
    api_key = "sk-" + secrets.token_hex(16)

    cfg = {
        "ntfy_topic": topic,
        "api_key": api_key,
        "hf_model_id": args.hf_model_id,
        "served_model_name": args.served_model_name,
        "weights_dataset": args.weights_dataset or "__hf_fp8_download__",
        "env_dataset": args.env_dataset or "__fp8_env_not_attached__",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "mtp_tokens": args.mtp,
        "reasoning_effort_default": args.reasoning_effort,
        "keepalive_min": args.keepalive_min,
        "text_only": args.text_only,
        "verbose": args.verbose,
        "fast_start": args.fast_start,
    }
    if args.no_tools:
        cfg["tool_call_parser"] = ""

    if args.fast_start and not args.env_dataset:
        say("WARNING: --fast-start without an FP8 XLA cache means first-use shapes compile cold.")

    wrapper = inject_cfg(WRAPPER_SRC.read_text(), cfg)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "serve_qwen38_fp8.py").write_text(wrapper)
        (td / "serve_qwen38.py").write_text(BASE_KERNEL_SRC.read_text())
        metadata = {
            "id": f"{user}/{args.slug}",
            "title": args.slug,
            "code_file": "serve_qwen38_fp8.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "true",
            "enable_internet": "true",
            "dataset_sources": dataset_sources(args),
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (td / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1))

        say(f"Pushing {user}/{args.slug} (TPU v5e-8, OrcaRouter FP8)...")
        r = kaggle("kernels", "push", "-p", str(td))
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out.lower():
            raise SystemExit("Kaggle push failed:\n" + out.strip())

    STATE_FILE.write_text(
        json.dumps(
            {
                "kernel": f"{user}/{args.slug}",
                "topic": topic,
                "api_key": api_key,
                "model": args.served_model_name,
            }
        )
    )
    say(
        "Pushed. The FP8 checkpoint is ~30.9 GB; without a Kaggle weight mirror "
        "it will download from Hugging Face before TPU compilation."
    )
    say(
        "HF access is read inside Kaggle from the secret named HF_TOKEN. "
        "The token is not embedded in the pushed kernel."
    )
    watch(f"{user}/{args.slug}", topic)


def read_events(topic, since):
    try:
        with urllib.request.urlopen(
            f"https://ntfy.sh/{topic}/json?poll=1&since={since}", timeout=15
        ) as response:
            body = response.read().decode()
    except Exception:
        return []
    events = []
    for line in body.splitlines():
        try:
            envelope = json.loads(line)
            if envelope.get("event") != "message":
                continue
            events.append((envelope["time"], json.loads(envelope.get("message", "{}"))))
        except Exception:
            continue
    return events


def render_event(event):
    phase = event.get("phase", "?")
    if phase == "weights-download":
        say(f"Downloading {MODEL_ID} from Hugging Face...")
    elif phase == "weights-downloaded":
        say(f"FP8 checkpoint downloaded in {event.get('secs', '?')} s.")
    elif phase == "server-launch":
        say("Starting vLLM and compiling TPU graphs...")
    elif phase == "compiling":
        say(f"Loading / compiling... {event.get('elapsed_s', 0) // 60} min elapsed.")
    elif phase == "tunnel-url":
        say(f"Endpoint reserved: {event.get('endpoint')} (wait for READY).")
    elif phase == "benchmark":
        say(
            f"Benchmark: {event.get('decode_tok_s', '?')} tok/s; "
            f"sanity={event.get('sanity', '')!r}"
        )
    elif phase == "ready":
        print("\n" + "=" * 68)
        print("  ORCAROUTER QWEN3.8 FP8 ENDPOINT IS LIVE")
        print(f"  base URL : {event.get('endpoint')}")
        print(f"  API key  : {event.get('api_key')}")
        print(f"  model    : {event.get('model')}")
        print(f"  context  : {event.get('max_model_len')}")
        print("=" * 68)
    elif phase == "failed":
        say(f"FAILED at {event.get('step', '?')}.")
        if event.get("tail"):
            print(event["tail"])
    elif phase == "heartbeat":
        say(f"Still serving ({event.get('up_min', '?')} min).")
    elif phase in ("installed", "mtp-patch-applied", "cache-missing", "serving"):
        say(phase.replace("-", " ").capitalize() + ".")
    elif phase not in ("install",):
        say(f"{phase}: " + json.dumps({k: v for k, v in event.items() if k != "phase"}))


def watch(kernel, topic):
    since = int(time.time()) - 600
    last_status = None
    try:
        while True:
            for ts, event in read_events(topic, since):
                since = max(since, ts)
                render_event(event)
                if event.get("phase") in ("failed", "auto-shutdown", "stopped"):
                    return
            r = kaggle("kernels", "status", kernel)
            output = (r.stdout or "") + (r.stderr or "")
            m = re.search(r'"KernelWorkerStatus\.(\w+)"', output)
            status = m.group(1) if m else "UNKNOWN"
            if status != last_status:
                say(f"Kaggle status: {status}")
                last_status = status
            if status in ("ERROR", "CANCELACKNOWLEDGED", "COMPLETE"):
                return
            time.sleep(30)
    except KeyboardInterrupt:
        say("Detached. The Kaggle kernel continues running; use `status -f` or `stop`.")


def load_state():
    if not STATE_FILE.exists():
        raise SystemExit("No FP8 launch state found; run `python launch_fp8.py serve` first.")
    return json.loads(STATE_FILE.read_text())


def cmd_status(args):
    state = load_state()
    r = kaggle("kernels", "status", state["kernel"])
    say(((r.stdout or "") + (r.stderr or "")).strip())
    for _, event in read_events(state["topic"], int(time.time()) - 24 * 3600)[-10:]:
        render_event(event)
    if args.follow:
        watch(state["kernel"], state["topic"])


def cmd_stop(_args):
    state = load_state()
    say(f"Deleting {state['kernel']} to terminate its TPU session...")
    p = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "delete", state["kernel"]],
        input="yes\n",
        capture_output=True,
        text=True,
    )
    say((p.stdout + p.stderr).strip() or "done")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("--user")
    serve.add_argument("--slug", default="qwen38-uncensored-fp8-tpu")
    serve.add_argument("--hf-model-id", default=MODEL_ID)
    serve.add_argument("--served-model-name", default=SERVED_MODEL_NAME)
    serve.add_argument(
        "--weights-dataset",
        default="",
        help="optional Kaggle dataset containing this FP8 checkpoint; otherwise download from HF",
    )
    serve.add_argument(
        "--env-dataset",
        default="",
        help="optional FP8-specific v5e-8 XLA/env bundle; do not use the BF16 bundle",
    )
    serve.add_argument("--max-model-len", type=int, default=262144)
    serve.add_argument("--max-num-seqs", type=int, default=4)
    serve.add_argument("--mtp", type=int, default=3)
    serve.add_argument(
        "--reasoning-effort", choices=("xhigh", "medium", "low"), default="xhigh"
    )
    serve.add_argument("--keepalive-min", type=int, default=480)
    serve.add_argument("--text-only", action="store_true")
    serve.add_argument("--no-tools", action="store_true")
    serve.add_argument("--verbose", action="store_true")
    serve.add_argument("--fast-start", action="store_true")
    serve.set_defaults(fn=cmd_serve)

    status = sub.add_parser("status")
    status.add_argument("--follow", "-f", action="store_true")
    status.set_defaults(fn=cmd_status)

    stop = sub.add_parser("stop")
    stop.set_defaults(fn=cmd_stop)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
