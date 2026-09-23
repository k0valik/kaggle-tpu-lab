import json, re, subprocess, sys, time, urllib.request
from pathlib import Path

TOPIC = "ktl-bonsai-6d2f8e1a4b7c3905"
KEY = "1e0afcc97b0ba77076ef35a63664d578"
ALIAS = "Ternary-Bonsai-2-27B-Abliterated"
PORT = 8080
GGUF_REPO = "Hikari07jp/Ternary-Bonsai-2-27B-Abliterated-GGUF"
MODEL_FILE = "Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf"  # 7.21 GB, ggml type 142
FORK = "PrismML-Eng/llama.cpp"  # custom ternary kernels; stock llama.cpp = garbage

def ntfy(msg):
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://ntfy.sh/{TOPIC}", data=msg.encode(), method="POST"), timeout=10)
    except Exception:
        pass

def say(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)
    ntfy(f"bonsai-t4: {msg}")

def detect_gpu():
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                        "--format=csv,noheader"], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
    return lines

def start_tunnel():
    cf = """
set -e
if [ ! -x /tmp/cloudflared ]; then
  echo "downloading cloudflared..."
  curl -sL --retry 3 -o /tmp/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /tmp/cloudflared
fi
"""
    p = subprocess.run(["bash", "-lc", cf], capture_output=True, text=True)
    sys.stdout.write((p.stdout + p.stderr)[-400:]); sys.stdout.flush()
    if p.returncode != 0:
        return ""
    for attempt in range(3):
        # NOTE: never `pkill -f cloudflared` here - the pattern matches this very
        # shell's command line and kills it before nohup runs (empty tunnel.log).
        subprocess.run(["bash", "-lc",
            "rm -f /tmp/logs/tunnel.log; pkill -x cloudflared 2>/dev/null; sleep 1; "
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
        try:
            tail = Path("/tmp/logs/tunnel.log").read_text(errors="replace")[-300:]
        except Exception:
            tail = "(no tunnel.log - launch shell died)"
        say(f"tunnel attempt {attempt + 1}/3 failed: {tail.strip()}")
    return ""

def go_live(URL):
    say(f"READY endpoint={URL}/v1 api_key={KEY} model={ALIAS}")
    test = subprocess.run(
        ["bash", "-lc",
         f"curl -s http://127.0.0.1:{PORT}/v1/chat/completions "
         "-H 'Content-Type: application/json' "
         f"-H 'Authorization: Bearer {KEY}' "
         "-d '{\"model\":\"" + ALIAS + "\",\"messages\":[{\"role\":\"user\","
         "\"content\":\"Say OK\"}],\"max_tokens\":8,\"chat_template_kwargs\":"
         "{\"reasoning_effort\":\"medium\"}}'"],
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

def install_fork():
    """Prebuilt PrismML fork first (zero build); source build as fallback."""
    say("step 1/4 install PrismML llama.cpp fork (prebuilt CUDA 12.4)")
    pick = f"""
import json, urllib.request
d = json.load(urllib.request.urlopen(
    "https://api.github.com/repos/{FORK}/releases/latest"))
for a in d["assets"]:
    if "linux-cuda-12.4-x64" in a["name"] and a["name"].endswith(".tar.gz"):
        print(a["browser_download_url"]); break
else:
    raise SystemExit("no linux-cuda-12.4 asset in latest release")
"""
    p = subprocess.run([sys.executable, "-c", pick], capture_output=True, text=True)
    url = p.stdout.strip()
    if p.returncode != 0 or not url.startswith("http"):
        say("release lookup failed: " + (p.stderr or p.stdout)[-200:]); return False
    say("release: " + url.rsplit("/", 1)[-1])
    dl = f"""
set -e
mkdir -p /tmp/prism/bin /tmp/logs
curl -sL --retry 3 -o /tmp/prism/fork.tar.gz "{url}"
tar -xzf /tmp/prism/fork.tar.gz -C /tmp/prism/bin --strip-components=1
/tmp/prism/bin/llama-server --version | head -1
"""
    p = subprocess.run(["bash", "-lc", dl], capture_output=True, text=True)
    sys.stdout.write((p.stdout + p.stderr)[-600:]); sys.stdout.flush()
    if p.returncode == 0:
        say("fork binary OK"); return True
    say("prebuilt failed - falling back to source build (~15 min)")
    build = f"""
set -e
rm -rf /tmp/llama-prism
git clone --depth 1 https://github.com/{FORK} /tmp/llama-prism 2>&1 | tail -1
cmake -S /tmp/llama-prism -B /tmp/llama-prism/build -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > /tmp/logs/cmake.log 2>&1
cmake --build /tmp/llama-prism/build --config Release -j$(nproc) \
  > /tmp/logs/build.log 2>&1
/tmp/llama-prism/build/bin/llama-server --version | head -1
"""
    p = subprocess.run(["bash", "-lc", build], capture_output=True, text=True)
    sys.stdout.write((p.stdout + p.stderr)[-800:]); sys.stdout.flush()
    if p.returncode != 0:
        say("BUILD FAILED - see /tmp/logs/{cmake,build}.log"); return False
    say("source build OK"); return True

START_SERVER = None

def main():
    global START_SERVER
    Path("/tmp/logs").mkdir(exist_ok=True)
    subprocess.run(["mkdir", "-p", "/tmp/models"])

    say("step 0/4 hardware detect")
    gpus = detect_gpu()
    if not gpus:
        say("NO GPU FOUND - this kernel needs T4x2 (or any CUDA GPU); "
            "Kaggle gave a CPU-only container. Re-push with GPU accelerator.")
        sys.exit(1)
    ndev = len(gpus)
    say(f"GPU(s): {ndev} x " + "; ".join(gpus))
    multi = ndev >= 2

    if not install_fork():
        sys.exit(1)

    say("step 2/4 download GGUF (7.21 GB, PQ2_0 ternary)")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=GGUF_REPO, allow_patterns=[MODEL_FILE],
                          local_dir="/tmp/models")
        size = (Path("/tmp/models") / MODEL_FILE).stat().st_size
        say(f"download OK ({size/1e9:.2f} GB)")
    except Exception as ex:
        say("download failed: " + str(ex)[-300:]); sys.exit(1)

    say("step 3/4 launch llama-server")
    BIN = "/tmp/prism/bin/llama-server"
    if not Path(BIN).exists():
        BIN = "/tmp/llama-prism/build/bin/llama-server"
    base = f"-sm layer" if multi else ""
    def _start():
        start = f"""
pkill -x llama-server 2>/dev/null; sleep 2
BIN={BIN}
MODEL="/tmp/models/{MODEL_FILE}"
# Dual T4: layer split across both cards; single T4: all on one.
# PQ2_0 = 7.21 GB, fits easily; 65536 ctx stays conservative for KV + vision slack.
# reasoning_effort=medium is required (xhigh default -> empty answers, weak abliteration).
for FLAGS in \\
  "{base} -ngl 99 -c 65536 -fa on" \\
  "{base} -ngl 99 -c 32768 -fa on" \\
  "-ngl 99 -c 32768 -fa on" \\
  "-ngl 0 -c 16384 -fa off"; do
  echo "trying: $FLAGS"
  nohup $BIN -m "$MODEL" -a "{ALIAS}" \\
    --host 127.0.0.1 --port {PORT} $FLAGS \\
    --jinja --no-webui --threads 4 \\
    --chat-template-kwargs '{{"reasoning_effort": "medium"}}' \\
    > /tmp/logs/server.log 2>&1 &
  PID=$!
  OK=0
  for i in $(seq 1 90); do
    sleep 2
    curl -s http://127.0.0.1:{PORT}/health 2>/dev/null | grep -q ok && OK=1 && break
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
    say(f"server UP on 127.0.0.1:{PORT} ({'dual-GPU layer split' if multi else 'single GPU'})")

    say("step 4/4 tunnel")
    URL = start_tunnel()
    if not URL:
        say("tunnel failed after 3 attempts"); sys.exit(1)
    go_live(URL)

if __name__ == "__main__":
    main()
