import json, re, subprocess, sys, time, urllib.request
from pathlib import Path

TOPIC = "ktl-ornith-3f9c2b7e51a04d68"
KEY = "1e0afcc97b0ba77076ef35a63664d578"
ALIAS = "Ornith-1.5-35B-A3B-CRACK"
PORT = 8080

# P100 path (Q4_K GGUF; 21.7 GB > 16 GB VRAM -> experts split with --n-cpu-moe)
GGUF_REPO = "dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF"
MODEL_FILE = "Ornith-1.5-35B-A3B-CRACK-Q4_K.gguf"

# TPU path (bf16 safetensors; vllm-tpu loads safetensors only)
TPU_REPO = "huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated"
VLLM_TPU_VERSION = "0.28.0"

def ntfy(msg):
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://ntfy.sh/{TOPIC}", data=msg.encode(), method="POST"), timeout=10)
    except Exception:
        pass

def say(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)
    ntfy(f"ornith-serve: {msg}")

def detect_hw():
    tpu = sorted(Path("/dev").glob("apex_*"))
    if tpu:
        return "tpu", len(tpu)
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                        "--format=csv,noheader"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return "gpu", r.stdout.strip()
    return None, None

# ---------------- shared: tunnel + self-test + keepalive ----------------

def start_tunnel():
    cf = """
set -e
if [ ! -x /tmp/cloudflared ]; then
  echo "downloading cloudflared..."
  curl -sL --retry 3 -o /tmp/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /tmp/cloudflared
fi
/tmp/cloudflared --version 2>&1 | head -1
"""
    p = subprocess.run(["bash", "-lc", cf], capture_output=True, text=True)
    sys.stdout.write((p.stdout + p.stderr)[-400:]); sys.stdout.flush()
    if p.returncode != 0:
        return ""
    for attempt in range(3):
        subprocess.run(["bash", "-lc",
            "pkill -x cloudflared 2>/dev/null; sleep 1; "
            "nohup /tmp/cloudflared tunnel --url http://127.0.0.1:8080 --no-autoupdate "
            "> /tmp/logs/tunnel.log 2>&1 &"], capture_output=True, text=True)
        for _ in range(45):
            time.sleep(2)
            try:
                log = Path("/tmp/logs/tunnel.log").read_text(errors="replace")
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
                if m:
                    return m.group(0)
            except Exception:
                pass
        say(f"tunnel attempt {attempt + 1}/3 failed - retrying")
    return ""

def go_live(URL):
    say(f"READY endpoint={URL}/v1 api_key={KEY} model={ALIAS}")
    test = subprocess.run(
        ["bash", "-lc",
         f"curl -s http://127.0.0.1:{PORT}/v1/chat/completions "
         "-H 'Content-Type: application/json' "
         f"-H 'Authorization: Bearer {KEY}' "
         "-d '{\"model\":\"" + ALIAS + "\",\"messages\":[{\"role\":\"user\","
         "\"content\":\"Say OK\"}],\"max_tokens\":8}'"],
        capture_output=True, text=True).stdout
    try:
        reply = json.loads(test)["choices"][0]["message"]["content"]
    except Exception:
        reply = "(no reply: " + test[:120] + ")"
    say("self-test reply: " + reply)
    i = 0
    while True:
        time.sleep(60); i += 1
        ok = subprocess.run(["bash", "-lc",
                             f"curl -s --max-time 5 http://127.0.0.1:{PORT}/health"],
                            capture_output=True, text=True).stdout
        if "ok" not in ok:
            say("health check failed - restarting server")
            START_SERVER()
            say("restart complete")
        if i % 15 == 0:
            say(f"keepalive {URL}/v1")

# ---------------- GPU path: P100 + llama.cpp (Q4_K, expert split) ----------------

def gpu_path(info):
    say("path: GPU (llama.cpp) - " + info)
    subprocess.run(["mkdir", "-p", "/tmp/logs", "/tmp/models"])
    say("step 1/4 build llama.cpp")
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap",
                        "--format=csv,noheader"], capture_output=True, text=True)
    CC = r.stdout.split(",")[0].strip().replace(".", "") if "," in r.stdout else "60"
    build = f"""
set -e
# Kaggle's image has no libcuda.so* anywhere (not even toolkit stubs), which breaks
# cmake's CUDA::cuda_driver imported target. GGML_CUDA_NO_VMM removes that link.
rm -rf /tmp/llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp 2>&1 | tail -1
cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build \
  -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES={CC} \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > /tmp/logs/cmake.log 2>&1
cmake --build /tmp/llama.cpp/build --config Release -j$(nproc) \
  > /tmp/logs/build.log 2>&1
/tmp/llama.cpp/build/bin/llama-server --version | head -1
"""
    p = subprocess.run(["bash", "-lc", build], capture_output=True, text=True)
    sys.stdout.write(p.stdout[-1500:]); sys.stdout.flush()
    if p.returncode != 0:
        say("BUILD FAILED - see /tmp/logs/cmake.log, build.log; kernel exits")
        sys.exit(1)
    say("build OK")

    say("step 2/4 download GGUF (~21.7 GB)")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=GGUF_REPO, allow_patterns=[MODEL_FILE],
                          local_dir="/tmp/models")
        size = (Path("/tmp/models") / MODEL_FILE).stat().st_size
        say(f"download OK ({size/1e9:.1f} GB)")
    except Exception as ex:
        say("download failed: " + str(ex)[-300:])
        sys.exit(1)

    say("step 3/4 launch llama-server")
    ram = Path("/proc/meminfo").read_text().split("\n")[0]
    say("system RAM: " + ram.split(":")[1].strip())  # experts offloaded here live
    global START_SERVER
    def _start():
        start = """
pkill -x llama-server 2>/dev/null; sleep 2
BIN=/tmp/llama.cpp/build/bin/llama-server
# Q4_K is 21.7 GB > 16 GB VRAM, so routed experts are split with --n-cpu-moe
# (first N layers keep their expert weights on CPU, mmap'd from local SSD).
# Highest-GPU config first; the first healthy one wins.
#   ncm 20: ~14.7 GB VRAM (fastest, borderline) -> ncm 30: ~10.5 GB -> ncm 40: ~6 GB
for FLAGS in \\
  "-c 262144 -fa on --cache-type-k q4_0 --cache-type-v q4_0 -ngl 99 --n-cpu-moe 20" \\
  "-c 262144 -fa on --cache-type-k q4_0 --cache-type-v q4_0 -ngl 99 --n-cpu-moe 30" \\
  "-c 262144 -fa on --cache-type-k q4_0 --cache-type-v q4_0 -ngl 99 --n-cpu-moe 40" \\
  "-c 131072 -fa on --cache-type-k q4_0 --cache-type-v q4_0 -ngl 99 --n-cpu-moe 40"; do
  echo "trying: $FLAGS"
  nohup $BIN -m "/tmp/models/Ornith-1.5-35B-A3B-CRACK-Q4_K.gguf" -a "$ALIAS" \\
    --host 127.0.0.1 --port 8080 $FLAGS \\
    --cache-reuse 256 --jinja --no-webui --threads 4 \\
    > /tmp/logs/server.log 2>&1 &
  PID=$!
  OK=0
  for i in $(seq 1 75); do
    sleep 2
    curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && OK=1 && break
    kill -0 $PID 2>/dev/null || break
  done
  if [ "$OK" = "1" ]; then echo "UP: $FLAGS"; exit 0; fi
  pkill -x llama-server 2>/dev/null; sleep 2
done
echo "SERVER FAILED"; tail -30 /tmp/logs/server.log; exit 1
"""
        p = subprocess.run(["bash", "-lc", start], capture_output=True, text=True)
        sys.stdout.write(p.stdout[-1200:]); sys.stdout.flush()
        return p.returncode == 0
    START_SERVER = _start
    if not _start():
        say("server failed to start"); sys.exit(1)
    say("server UP on 127.0.0.1:8080")

    say("step 4/4 tunnel")
    URL = start_tunnel()
    if not URL:
        say("tunnel failed after 3 attempts"); sys.exit(1)
    go_live(URL)

# ---------------- TPU path: v5e-8 + vllm-tpu (bf16 safetensors) ----------------

def tpu_path(ndev):
    say(f"path: TPU detected ({ndev} device(s))")
    if ndev != 8:
        say(f"expected 8 devices for v5e-8, found {ndev} - continuing anyway")
    say("NOTE: qwen3_5_moe support in vllm-tpu 0.28.0 is unverified; if the engine "
        "rejects the architecture this kernel exits with the error in the log.")
    import os
    os.environ["HF_HOME"] = "/tmp/hf"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    subprocess.run(["mkdir", "-p", "/tmp/logs"])
    du = __import__("shutil").disk_usage("/tmp")
    say(f"/tmp free space: {du.free/1e9:.0f} GB (need ~75 GB for bf16 weights)")
    if du.free < 75e9:
        say("WARNING: low disk - download may fail")

    say("step 1/3 build venv with vllm-tpu (pinned)")
    install = f"""
set -e
{sys.executable} -m pip install -q uv > /tmp/logs/uv.log 2>&1
{sys.executable} -m uv venv /tmp/venv --python {sys.executable} -q
{sys.executable} -m uv pip install --python /tmp/venv/bin/python --torch-backend=cpu \
  "vllm-tpu=={VLLM_TPU_VERSION}" huggingface_hub >> /tmp/logs/uv.log 2>&1
/tmp/venv/bin/python -c "import vllm, tpu_inference; print('vllm-tpu OK')"
"""
    p = subprocess.run(["bash", "-lc", install], capture_output=True, text=True)
    sys.stdout.write((p.stdout + p.stderr)[-800:]); sys.stdout.flush()
    if p.returncode != 0:
        say("venv/install failed - see /tmp/logs/uv.log; kernel exits")
        sys.exit(1)
    say("venv OK")

    say("step 2/3 download bf16 weights (~70 GB, 16 shards)")
    dl = """
import time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(repo_id="huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated",
                      local_dir="/tmp/model")
print("downloaded to", p, f"in {time.time()-t0:.0f}s")
"""
    p = subprocess.run(["/tmp/venv/bin/python", "-c", dl], capture_output=True, text=True)
    sys.stdout.write((p.stdout + p.stderr)[-600:]); sys.stdout.flush()
    if p.returncode != 0:
        say("download failed; kernel exits"); sys.exit(1)
    say("download OK")

    say("step 3/3 launch vLLM (TP=8, 262k ctx, text-only, no MTP)")
    global START_SERVER
    def _start():
        args = ["/tmp/venv/bin/python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", "/tmp/model",
                "--tensor-parallel-size", "8",
                "--max-model-len", "262144",
                "--max-num-seqs", "4",
                "--port", str(PORT),
                "--api-key", KEY,
                "--served-model-name", ALIAS,
                "--reasoning-parser", "qwen3",
                "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
        log = open("/tmp/logs/vllm.log", "ab")
        proc = subprocess.Popen(args, stdout=log, stderr=log)
        for _ in range(2250):  # up to 75 min: cold XLA compile, no cache for this arch
            time.sleep(2)
            try:
                h = urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/health", timeout=3)
                if h.status == 200:
                    return True
            except Exception:
                if proc.poll() is not None:
                    tail = Path("/tmp/logs/vllm.log").read_text(errors="replace").splitlines()[-25:]
                    sys.stdout.write("\n".join(tail) + "\n"); sys.stdout.flush()
                    return False
        return False
    START_SERVER = _start
    if not _start():
        say("vLLM failed to become healthy (see /tmp/logs/vllm.log)")
        say("If the error is 'architecture ... not supported', qwen3_5_moe needs a "
            "newer vllm-tpu; bump VLLM_TPU_VERSION and retry.")
        sys.exit(1)
    say("server UP on 127.0.0.1:8080")

    URL = start_tunnel()
    if not URL:
        say("tunnel failed after 3 attempts"); sys.exit(1)
    go_live(URL)

# ---------------- main ----------------

START_SERVER = None
Path("/tmp/logs").mkdir(exist_ok=True)
say("step 0/4 hardware detect")
hw, info = detect_hw()
say(f"hardware: {hw} ({info})")
if hw == "tpu":
    tpu_path(info)
elif hw == "gpu":
    gpu_path(info)
else:
    say("NO ACCELERATOR FOUND - no /dev/apex_*, no nvidia-smi. Kaggle likely gave a "
        "CPU-only container (unverified account or fleet pressure). Phone-verify the "
        "account and retry, or push with GPU/TPU accelerator selected.")
    sys.exit(1)
