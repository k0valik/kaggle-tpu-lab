#!/usr/bin/env python3
"""Generate the public dual-T4 Kaggle notebook from its canonical server."""

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebook" / "gpu" / "qwen38-t4x2-serve.ipynb"
SERVER = REPO / "kernel" / "serve_qwen38_gpu.py"
SETTINGS_IMAGE = (
    "https://raw.githubusercontent.com/Kitkitkittt/kaggle-tpu-lab/main/"
    "notebook/assets/kaggle-settings-internet-accelerator.png"
)


INTRO = """# Run Qwen3.8-27B on free Kaggle GPU T4 ×2

## Beginner guide: quantized Q4 model + temporary OpenAI-compatible API

This is the **GPU alternative** to the repository's TPU v5e-8 notebook. Two
Tesla T4 GPUs provide about 32 GiB of combined VRAM, so this notebook uses the
`Qwen3.8-27B-UD-Q4_K_M.gguf` quantization (about 15.3 GiB). The TPU notebook
uses the much larger bf16 weights; the two paths are intentionally separate.

What this notebook does:

1. Confirms that Kaggle attached two T4 GPUs and has enough scratch space.
2. Downloads checksum-pinned CUDA llama.cpp binaries and the Q4 GGUF model.
3. Splits model layers evenly across both GPUs.
4. Starts an API-key-protected OpenAI-compatible server.
5. Opens a temporary Cloudflare Quick Tunnel and performs a local self-test.
6. Automatically stops after 10 minutes unless you change the configuration.

Expect a roughly 16 GB model download on every fresh session. The first run can
take 10–25 minutes depending on Hugging Face and Kaggle network speed.

> **Security:** Never put your Kaggle token in a notebook. This notebook creates
> a temporary inference key for each run. Do not publish the live URL/key output,
> and stop the session when finished.
"""


SETUP = f"""## Step 1 — Copy the notebook and select GPU T4 ×2

The public notebook is read-only. Sign in, click **Copy & Edit**, then:

1. Open **Settings** in the notebook menu.
2. If the menu says **Turn off internet**, Internet is already **on**. Otherwise,
   click the Internet item to turn it on.
3. Choose **Accelerator → GPU T4 ×2**. Do not choose P100 or TPU for this notebook.
4. Let Kaggle restart the session after changing the accelerator.
5. Run the next hardware-check cell before using **Run All**.

![Kaggle Settings menu showing Internet enabled and GPU T4 x2]({SETTINGS_IMAGE})

The check must list two Tesla T4 devices. One GPU, a P100, a CPU-only session,
or a TPU cannot run this specific two-GPU configuration.
"""


GPU_CHECK = '''# STEP 1 CHECK — verify two T4 GPUs and enough scratch space
import os
import shutil
import subprocess
from pathlib import Path

_result = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    text=True,
)
_gpus = [line.strip() for line in _result.stdout.splitlines() if line.strip()]
print("GPU inventory:")
for _gpu in _gpus:
    print("  ", _gpu)

if _result.returncode != 0 or len(_gpus) != 2 or any("T4" not in row.upper() for row in _gpus):
    raise RuntimeError(
        "Expected two Tesla T4 GPUs. Open Settings, turn Internet on, choose "
        "Accelerator -> GPU T4 x2, restart the session, and rerun this cell."
    )

_scratch = Path(os.environ.get("QWEN38_SCRATCH", "/tmp/qwen38-gpu"))
_scratch.mkdir(parents=True, exist_ok=True)
_free_gib = shutil.disk_usage(_scratch).free / 1024**3
print(f"Scratch space: {_free_gib:.1f} GiB free")
if _free_gib < 20:
    raise RuntimeError(f"At least 20 GiB of free scratch space is required at {_scratch}.")

print("✅ Hardware ready — two T4 GPUs and enough scratch space are available.")
'''


CONFIG_DOC = """## Step 2 — Use the safe first-run configuration

Run the next cell without editing it for your first successful launch:

- `ctx_size: 32768` gives a useful 32k-token context without pushing T4 memory.
- `parallel: 2` provides two server request slots.
- `mtp_tokens: 0` disables experimental MTP for the reliability-first run.
- `keepalive_min: 10` stops the server ten minutes after `READY` to conserve quota.

After the basic path works, you can increase `keepalive_min`. Change one setting
at a time; larger contexts consume more GPU memory.
"""


CONFIG = '''%%writefile gpu_serve_config.json
{
  "ctx_size": 32768,
  "parallel": 2,
  "mtp_tokens": 0,
  "keepalive_min": 10
}
'''


DOWNLOADS = """## Step 3 — Understand the downloads

The launch downloads two pinned artifacts into `/tmp/qwen38-gpu`:

- CUDA llama.cpp v0.4.0 binaries: about 146 MB.
- `Qwen3.8-27B-UD-Q4_K_M.gguf`: about 15.3 GiB.

Both files are verified with SHA-256 before execution. `/tmp/qwen38-gpu` is used so
the model does not consume the notebook's smaller persisted-output allowance.
Fresh Kaggle sessions lose these scratch files, so a later session downloads
them again.
"""


SERVER_NOTE = """## Step 4 — Prepare the reviewed GPU server

Run the next generated cell once. It writes the canonical GPU server from the
public repository. You do not need to edit the large cell; it is hidden when
Kaggle honors Jupyter's hidden-input metadata.
"""


CLIENT_GUIDE = """## Step 5 — Keep this API test ready

The final Kaggle cell stays active while serving, so run the test from another
terminal, local Python session, Colab notebook, or API client. Wait for:

```text
READY — Qwen3.8-27B Q4 is live on both T4 GPUs
ENDPOINT: https://example.trycloudflare.com/v1
API KEY : sk-...
MODEL   : qwen3.8-27b-q4
```

Then replace the placeholders below with values from **your current run**:

```bash
curl "YOUR_ENDPOINT/chat/completions" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen3.8-27b-q4",
    "messages": [{"role": "user", "content": "Explain dual-GPU inference briefly."}],
    "max_tokens": 128,
    "reasoning_effort": "none"
  }'
```

The printed endpoint already includes `/v1`; do not add it twice. Remove
`reasoning_effort` later if you want the model's reasoning mode.
"""


TROUBLESHOOTING = """## Troubleshooting

| Symptom | Fix |
|---|---|
| Hardware check lists CPU or one GPU | Stop the session, select **GPU T4 ×2**, and rerun the check. |
| It lists P100 | Change to T4 ×2; the pinned CUDA binary and memory plan target T4. |
| Download fails | Confirm Internet is on, then rerun; the downloader resumes partial files. |
| Checksum mismatch | Rerun so the bad partial file is removed and downloaded again. |
| `llama-server` exits or reports CUDA OOM | Keep the default 32k context, stop other GPU processes, and restart the session. |
| Tunnel gives 502 | Wait for the `READY` banner; 502 is expected while the model loads. |
| API returns 401 | Use the generated inference key from the current run—not a Kaggle token or old key. |
| Endpoint disappears | The 10-minute keepalive elapsed or the Kaggle session stopped; launch again. |

Use Kaggle's **Stop session** control when finished. The tunnel and generated
key stop working when the session ends.
"""


LAUNCH = """## Step 6 — Launch the model server

Run the final cell and leave it active. Expected phases are:

1. `gpu-ready` — two T4 GPUs were detected.
2. `download` — pinned binaries/model are being fetched or verified.
3. `model-ready` — the 15.3 GiB GGUF passed checksum and size checks.
4. `server-starting` / `model-loading` — layers are loading across both GPUs.
5. `tunnel-url` — the temporary URL exists but may still return 502.
6. `self-test` and `ready` — copy the endpoint/key and use the Step 5 test.

The active cell is the server. It exits automatically after the configured
keepalive period and always attempts to stop the server and tunnel processes.
"""


def source(text):
    return text.splitlines(keepends=True)


def markdown(cell_id, text):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source(text)}


def code(cell_id, text, metadata=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": metadata or {},
        "outputs": [],
        "source": source(text),
    }


def build_notebook():
    server = SERVER.read_text()
    if not server.endswith("\n"):
        server += "\n"
    return {
        "cells": [
            markdown("gpu-introduction", INTRO),
            markdown("gpu-kaggle-setup", SETUP),
            code("gpu-check", GPU_CHECK),
            markdown("gpu-config-help", CONFIG_DOC),
            code("gpu-configuration", CONFIG),
            markdown("gpu-downloads", DOWNLOADS),
            markdown("gpu-server-source", SERVER_NOTE),
            code(
                "gpu-embedded-server",
                "%%writefile serve_qwen38_gpu.py\n" + server,
                {"jupyter": {"source_hidden": True}, "tags": ["hide-input"]},
            ),
            markdown("gpu-client-guide", CLIENT_GUIDE),
            markdown("gpu-troubleshooting", TROUBLESHOOTING),
            markdown("gpu-launch-help", LAUNCH),
            code("gpu-launch-server", "!python serve_qwen38_gpu.py\n"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render():
    return json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the notebook is out of date")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not NOTEBOOK.exists() or NOTEBOOK.read_text() != expected:
            print(f"{NOTEBOOK.relative_to(REPO)} is out of date", file=sys.stderr)
            return 1
        print(f"{NOTEBOOK.relative_to(REPO)} is synchronized")
        return 0
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(expected)
    print(f"wrote {NOTEBOOK.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
