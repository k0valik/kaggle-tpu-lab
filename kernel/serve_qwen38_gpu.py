#!/usr/bin/env python3
"""Serve a quantized Qwen3.8-27B GGUF on Kaggle's two Tesla T4 GPUs.

This is the GPU companion to ``serve_qwen38.py``. It deliberately uses a
Q4_K_M GGUF instead of the TPU notebook's bf16 safetensors: two T4s provide
32 GiB of combined VRAM, which is not enough for the roughly 55 GiB bf16 model.

The script downloads pinned CUDA llama.cpp binaries and the pinned GGUF into
Kaggle scratch space, starts an authenticated OpenAI-compatible server split
across both GPUs, creates a temporary Cloudflare Quick Tunnel, runs a short
self-test, and shuts everything down after ``keepalive_min``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULTS = {
    "model_repo": "unsloth/Qwen3.8-27B-GGUF",
    "model_revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
    "model_file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
    "model_sha256": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
    "model_size": 16_464_440_224,
    "llama_url": (
        "https://github.com/ai-dock/llama.cpp-cuda/releases/download/v0.4.0/"
        "llama.cpp-v0.4.0-cuda-12.8-amd64.tar.gz"
    ),
    "llama_sha256": "7a229ac0357d9d80931e6651a4231ee052157875dea9e6160b4636763b5a4bad",
    "cloudflared_url": (
        "https://github.com/cloudflare/cloudflared/releases/download/2026.8.3/"
        "cloudflared-linux-amd64"
    ),
    "cloudflared_sha256": "f29324fe934d1e100617484c78deef803c4dc2cd351d645bbde42e96b4fccc5e",
    "ctx_size": 32768,
    "parallel": 2,
    "mtp_tokens": 0,
    "keepalive_min": 10,
    "api_key": "",
    "served_model_name": "qwen3.8-27b-q4",
}

CFG = dict(DEFAULTS)
config_path = Path("gpu_serve_config.json")
if config_path.exists():
    CFG.update(json.loads(config_path.read_text()))
if not CFG["api_key"]:
    CFG["api_key"] = "sk-" + secrets.token_hex(16)

PORT = 8000
SCRATCH = Path(os.environ.get("QWEN38_SCRATCH", "/tmp/qwen38-gpu"))
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else SCRATCH
BIN_ARCHIVE = SCRATCH / "llama.cpp-v0.4.0-cuda-12.8-amd64.tar.gz"
BIN_DIR = SCRATCH / "llama.cpp-v0.4.0"
CLOUDFLARED = SCRATCH / "cloudflared-2026.8.3"
MODEL_PATH = SCRATCH / CFG["model_file"]
SERVER_LOG = WORK / "llama-server.log"
TUNNEL_LOG = WORK / "cloudflared.log"
T0 = time.time()


def log(*parts):
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def publish(phase, **details):
    log("PHASE", phase, json.dumps(details, sort_keys=True) if details else "")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checked(url: str, target: Path, expected_sha256: str):
    """Resume a pinned download and verify it before use."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        log("Checking cached", target.name)
        if file_sha256(target) == expected_sha256:
            log("✅ Reusing verified", target.name)
            return
        target.unlink()

    partial = target.with_suffix(target.suffix + ".part")
    publish("download", file=target.name)
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "5", "--retry-all-errors", "--continue-at", "-",
        "--output", str(partial), url,
    ]
    subprocess.run(command, check=True)
    actual = file_sha256(partial)
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {target.name}: expected {expected_sha256}, got {actual}"
        )
    partial.replace(target)
    log("✅ Downloaded and verified", target.name)


def gpu_inventory():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append({"index": fields[0], "name": fields[1], "memory_mib": int(fields[2])})
    return rows


def is_dual_t4(rows) -> bool:
    return len(rows) == 2 and all("T4" in row["name"].upper() for row in rows)


def require_dual_t4():
    rows = gpu_inventory()
    log("GPU inventory:", json.dumps(rows))
    if not is_dual_t4(rows):
        raise RuntimeError(
            "Expected two Tesla T4 GPUs. In Kaggle Settings, turn Internet on, "
            "choose Accelerator -> GPU T4 x2, restart the session, and rerun."
        )
    publish("gpu-ready", devices=rows)


def safe_extract(archive: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination)


def prepare_llama_server():
    download_checked(CFG["llama_url"], BIN_ARCHIVE, CFG["llama_sha256"])
    marker = BIN_DIR / ".verified"
    if not marker.exists():
        if BIN_DIR.exists():
            shutil.rmtree(BIN_DIR)
        safe_extract(BIN_ARCHIVE, BIN_DIR)
        marker.touch()
    servers = list(BIN_DIR.rglob("llama-server"))
    if not servers:
        raise RuntimeError("The verified llama.cpp archive contains no llama-server binary")
    server = servers[0]
    server.chmod(0o755)
    lib_dirs = sorted({str(path.parent) for path in BIN_DIR.rglob("*.so*")})
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")])
    version = subprocess.run([str(server), "--version"], env=env, capture_output=True, text=True)
    if version.returncode != 0:
        raise RuntimeError("llama-server could not start: " + (version.stderr or version.stdout)[-800:])
    log("✅", (version.stdout or version.stderr).strip().splitlines()[0])
    return server, env


def prepare_cloudflared():
    download_checked(CFG["cloudflared_url"], CLOUDFLARED, CFG["cloudflared_sha256"])
    CLOUDFLARED.chmod(0o755)


def download_model():
    free_gib = shutil.disk_usage(SCRATCH).free / 1024**3
    log(f"Scratch space available: {free_gib:.1f} GiB")
    if free_gib < 20 and not MODEL_PATH.exists():
        raise RuntimeError(f"At least 20 GiB of free scratch space is required at {SCRATCH}")
    model_url = (
        f"https://huggingface.co/{CFG['model_repo']}/resolve/"
        f"{CFG['model_revision']}/{CFG['model_file']}"
    )
    log("The quantized model is about 15.3 GiB; this is the longest download.")
    download_checked(model_url, MODEL_PATH, CFG["model_sha256"])
    if MODEL_PATH.stat().st_size != CFG["model_size"]:
        raise RuntimeError(
            f"Unexpected model size: expected {CFG['model_size']}, got {MODEL_PATH.stat().st_size}"
        )
    publish("model-ready", path=str(MODEL_PATH), bytes=MODEL_PATH.stat().st_size)


def tail(path: Path, size=4000):
    try:
        return path.read_text(errors="replace")[-size:]
    except FileNotFoundError:
        return "(log file not created)"


def server_command(server: Path):
    command = [
        str(server),
        "--model", str(MODEL_PATH),
        "--alias", CFG["served_model_name"],
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--api-key", CFG["api_key"],
        "--ctx-size", str(CFG["ctx_size"]),
        "--parallel", str(CFG["parallel"]),
        "--n-gpu-layers", "all",
        "--split-mode", "layer",
        "--tensor-split", "1,1",
    ]
    if int(CFG["mtp_tokens"]) > 0:
        command += [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(CFG["mtp_tokens"]),
            "--spec-draft-p-min", "0.7",
        ]
    return command


def api_request(path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {CFG['api_key']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_server(process, timeout=1200):
    started = time.time()
    next_update = 0
    while time.time() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {process.returncode}:\n{tail(SERVER_LOG)}"
            )
        try:
            api_request("/v1/models", timeout=5)
            return int(time.time() - started)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        elapsed = int(time.time() - started)
        if elapsed >= next_update:
            publish("model-loading", elapsed_s=elapsed)
            next_update += 60
        time.sleep(5)
    raise RuntimeError("llama-server did not become healthy within 20 minutes:\n" + tail(SERVER_LOG))


def start_tunnel():
    handle = TUNNEL_LOG.open("w")
    process = subprocess.Popen(
        [
            str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{PORT}",
            "--no-autoupdate", "--protocol", "quic",
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + 180
    while time.time() < deadline:
        if process.poll() is not None:
            break
        match = pattern.search(tail(TUNNEL_LOG, 12000))
        if match:
            return process, handle, match.group(0).rstrip("/")
        time.sleep(2)
    return process, handle, None


def stop_process(process, label):
    if process is None or process.poll() is not None:
        return
    log("Stopping", label)
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def self_test():
    body = {
        "model": CFG["served_model_name"],
        "messages": [{"role": "user", "content": "Reply with exactly: GPU ready"}],
        "max_tokens": 64,
        "temperature": 0,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = api_request("/v1/chat/completions", body, timeout=180)
    message = response["choices"][0]["message"]
    answer = (message.get("content") or message.get("reasoning_content") or "").strip()
    if not answer:
        raise RuntimeError("The local API self-test returned an empty response")
    publish("self-test", answer=answer[:100])
    return answer


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    require_dual_t4()
    server_process = tunnel_process = None
    server_handle = tunnel_handle = None
    try:
        publish("setup")
        server_binary, server_env = prepare_llama_server()
        prepare_cloudflared()
        download_model()

        publish(
            "server-starting",
            ctx_size=CFG["ctx_size"],
            parallel=CFG["parallel"],
            mtp_tokens=CFG["mtp_tokens"],
        )
        server_handle = SERVER_LOG.open("w")
        server_process = subprocess.Popen(
            server_command(server_binary),
            env=server_env,
            stdout=server_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        tunnel_process, tunnel_handle, public_url = start_tunnel()
        if public_url:
            publish("tunnel-url", endpoint=public_url + "/v1", note="wait for READY")
        else:
            publish("tunnel-failed", log=tail(TUNNEL_LOG, 1000))

        startup_seconds = wait_for_server(server_process)
        self_test()
        endpoint = public_url + "/v1" if public_url else "http://127.0.0.1:8000/v1"
        log("#" * 72)
        log("# READY — Qwen3.8-27B Q4 is live on both T4 GPUs")
        log("# ENDPOINT:", endpoint)
        log("# API KEY :", CFG["api_key"])
        log("# MODEL   :", CFG["served_model_name"])
        log("#" * 72)
        publish(
            "ready",
            endpoint=endpoint if public_url else None,
            api_key=CFG["api_key"],
            model=CFG["served_model_name"],
            startup_seconds=startup_seconds,
        )

        deadline = time.time() + int(CFG["keepalive_min"]) * 60
        while time.time() < deadline:
            if server_process.poll() is not None:
                raise RuntimeError("llama-server stopped unexpectedly:\n" + tail(SERVER_LOG))
            time.sleep(15)
        publish("complete", reason="keepalive elapsed")
    except KeyboardInterrupt:
        publish("complete", reason="notebook interrupted")
    except Exception as error:
        publish("failed", error=str(error)[-2000:])
        raise
    finally:
        stop_process(tunnel_process, "Cloudflare tunnel")
        stop_process(server_process, "llama-server")
        if tunnel_handle:
            tunnel_handle.close()
        if server_handle:
            server_handle.close()


if __name__ == "__main__":
    main()
