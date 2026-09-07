#!/usr/bin/env python3
"""Generate the public Kaggle notebook from the canonical serving script.

Run ``python tools/sync_notebook.py`` after editing ``kernel/serve_qwen38.py``.
CI and reviewers can use ``python tools/sync_notebook.py --check``.
"""
import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebook" / "qwen38-tpu-serve.ipynb"
SERVER = REPO / "kernel" / "serve_qwen38.py"


INTRO = """# Run Qwen3.8-27B on a free Kaggle TPU v5e-8

## Beginner-friendly guide: from a blank session to an OpenAI-compatible API

This notebook runs the full **Qwen3.8-27B bf16** model with vLLM across all eight
chips of a Kaggle TPU v5e-8. When startup finishes, it prints a temporary public
URL and API key that work with `curl`, the OpenAI Python client, and compatible
coding tools.

### What you need

- A Kaggle account that can use TPUs. Kaggle may require phone verification.
- Available TPU quota and no other TPU session running on the same account.
- About 5–20 minutes for startup, depending on cache availability. These timing
  estimates come from the original project; this fork does not claim a new
  successful benchmark until Kaggle provisions its TPU correctly.

### What this guide will do

1. Make an editable copy and select the correct Kaggle hardware.
2. Confirm that Python sees eight TPU devices.
3. Confirm that both large input datasets are attached.
4. Write a safe beginner configuration.
5. Start the model server and wait for the `READY` banner.
6. Test the endpoint from another terminal or application.

> **Security:** Never paste a Kaggle access token into a notebook cell. This
> notebook generates a different, temporary inference API key on every run.
> Stop the Kaggle session when finished and do not publish cell outputs that
> contain the endpoint or generated key.

Reviewed source: [Kitkitkittt/kaggle-tpu-lab](https://github.com/Kitkitkittt/kaggle-tpu-lab),
based on the MIT-licensed [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab).
"""


SETUP = """## Step 1 — Make an editable copy and configure Kaggle

The public notebook page is read-only. Do this before running any cell:

1. Sign in to Kaggle.
2. Click **Copy & Edit** near the top-right of this page.
3. In the editor's right sidebar, open **Session options**.
4. Set **Accelerator** to **TPU VM v5e-8**.
5. Turn **Internet** on. The temporary Cloudflare tunnel needs it.
6. Open the **Input** section and confirm that both of these appear:
   - `rahim3/qwen3-8-27b-bf16` — about 55 GB of model weights
   - `rahim3/qwen38-tpu-env-v5e8` — the matching XLA cache and helper binary

If an input is missing, click **+ Add Input**, search for the exact identifier,
and add it. Copied notebooks normally inherit the inputs, but verify them anyway.

![Kaggle Settings menu showing Internet enabled and the accelerator choices](https://raw.githubusercontent.com/Kitkitkittt/kaggle-tpu-lab/main/notebook/assets/kaggle-settings-internet-accelerator.png)

In the first panel, **Turn off internet** means Internet is already **on**. In
the second panel, select **TPU v5e-8** for this notebook. The **GPU T4 x2** item
is for the separate GPU notebook linked from the repository README.

Now run the next cell only. It is a fast hardware check; do not use **Run All**
until it prints the green success message.
"""


TPU_CHECK = '''# STEP 1 CHECK — confirm that Kaggle attached the requested TPU
import jax

_devices = jax.devices()
_device_rows = [
    f"{device.platform}:{device.id} ({getattr(device, 'device_kind', 'unknown')})"
    for device in _devices
]
print("JAX devices:", ", ".join(_device_rows))

if len(_devices) != 8 or any(device.platform != "tpu" for device in _devices):
    raise RuntimeError(
        "Kaggle did not attach a TPU v5e-8 to this session. "
        "Stop the session, open Session options, select TPU VM v5e-8, "
        "check account verification/quota, and run this cell again."
    )

print("✅ TPU ready — JAX can see all 8 TPU devices.")
'''


INPUTS = """## Step 2 — Verify the two attached datasets

The model is too large to redownload casually. This check prevents a long run
from starting with missing inputs. Run it after the TPU check passes.
"""


INPUT_CHECK = '''# STEP 2 CHECK — confirm that the weights and TPU cache are mounted
from pathlib import Path

_input_root = Path("/kaggle/input")
_expected_inputs = {
    "qwen3-8-27b-bf16": "Qwen3.8-27B bf16 model weights",
    "qwen38-tpu-env-v5e8": "matching TPU environment and XLA cache",
}
_missing = []

for _slug, _description in _expected_inputs.items():
    _matches = list(_input_root.glob(_slug)) + list(
        _input_root.glob(f"datasets/*/{_slug}")
    )
    if _matches:
        print(f"✅ {_description}: {_matches[0]}")
    else:
        print(f"❌ Missing {_description}: {_slug}")
        _missing.append(_slug)

if _missing:
    raise RuntimeError(
        "Missing Kaggle inputs: " + ", ".join(_missing) +
        ". Use + Add Input in the right sidebar, then rerun this cell."
    )

print("✅ Inputs ready — no model download is needed.")
'''


CONFIG = """%%writefile serve_config.json
{
  "max_model_len": 262144,
  "max_num_seqs": 4,
  "mtp_tokens": 3,
  "reasoning_effort_default": "xhigh",
  "text_only": true,
  "fast_start": true,
  "keepalive_min": 30
}
"""


CONFIG_DOC = """## Step 3 — Choose the beginner configuration

The next cell writes `serve_config.json`. The defaults are designed for a first
text-only run, so you can run the cell without editing anything:

- `max_model_len: 262144` keeps the model's native context window.
- `max_num_seqs: 4` allows up to four simultaneous requests.
- `mtp_tokens: 3` enables the bundled, verified state-rollback patch for MTP
  speculative decoding. Leave it at `3` for the documented path.
- `text_only: true` skips the vision tower, reducing startup work. Change it to
  `false` only after the text path works and you need image inputs.
- `fast_start: true` makes the endpoint available before every request shape is
  warmed. The first unusual request can pause for compilation.
- `keepalive_min: 30` automatically ends serving after 30 minutes so a forgotten
  notebook does not keep consuming quota. Increase it when you need more time.

Advanced throughput option: use `131072` with `max_num_seqs: 16`. The bundled
cache covers both that profile and the beginner `262144`/`4` profile. Other
combinations may compile from scratch and take longer.
"""


SYNTHESIS = """## What is happening under the hood? (optional reading)

Two reviewed input notebooks run quantized GGUF files through CUDA/llama.cpp on
dual T4 GPUs. Their download and chat ideas are useful GPU fallbacks, but CUDA
binaries, GPU layer offload, and `tensor_split` do not apply to a TPU VM.

This notebook instead follows the current TPU-native implementation: bf16
safetensors, eight-way TPU tensor parallelism, vLLM TPU, a matching XLA compile
cache, generated authentication, and automatic shutdown. The full comparison
and the items deliberately not copied are documented in
[`docs/INPUT_AUDIT.md`](https://github.com/Kitkitkittt/kaggle-tpu-lab/blob/main/docs/INPUT_AUDIT.md).
"""


SERVER_NOTE = """## Step 4 — Prepare the reviewed serving program

Run the next cell once. It writes the canonical serving script from the public
repository into this Kaggle session. You do not need to read or edit the large
generated cell: it is hidden when the notebook UI honors hidden-input metadata.

Keeping this cell generated from `kernel/serve_qwen38.py` prevents the beginner
guide and the reviewed runtime source from silently drifting apart.
"""


CLIENT_GUIDE = """## Step 5 — Save these test instructions before starting

The final cell stays busy while the server is running, so test from a **separate
terminal, Colab notebook, local Python session, or API client**.

Wait until the Kaggle output contains a banner like this:

```text
READY — the server is live
ENDPOINT : https://example.trycloudflare.com/v1
API KEY  : sk-...
MODEL    : qwen3.8-27b
```

Copy your own `ENDPOINT` and `API KEY` values. Do not copy the placeholders below.

### Test with curl

```bash
curl "YOUR_ENDPOINT/chat/completions" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Explain TPUs in three sentences."}],
    "chat_template_kwargs": {"reasoning_effort": "low"}
  }'
```

`YOUR_ENDPOINT` must include the printed `/v1`, but do not add a second `/v1`.

### Test with the OpenAI Python client

```python
# Run this on your computer or in a different notebook.
# pip install openai
from openai import OpenAI

client = OpenAI(base_url="YOUR_ENDPOINT", api_key="YOUR_API_KEY")
response = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Hello from Kaggle TPU!"}],
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)
print(response.choices[0].message.content)
```
"""


TROUBLESHOOTING = """## Troubleshooting before you launch

| Symptom | Meaning and fix |
|---|---|
| TPU check shows only `cpu:0` | Stop the session, explicitly select **TPU VM v5e-8**, and check phone verification, quota, and other active TPU sessions. |
| Dataset check says an input is missing | Use **+ Add Input** and add the exact missing identifier, then rerun the check. |
| Accelerator option is unavailable | The account may need verification, may have exhausted TPU quota, or may already have a TPU session. |
| Tunnel returns HTTP 502 | Normal before `READY`; keep waiting. If the server later fails, inspect the last `PHASE failed` line. |
| Startup appears paused | TPU compilation can be quiet for several minutes. Follow the numbered `STEP` and `PHASE` log lines. |
| First unusual request is slow | With `fast_start: true`, its shape may compile once; retry after it finishes. |
| API returns 401 | Use the generated inference key from the current run, not your Kaggle token or an older run's key. |
| Kaggle says the version completed but no `READY` appeared | The run failed or auto-stopped; Kaggle's version status alone does not prove the model served successfully. |

### How to stop safely

Use Kaggle's **Stop session** control when finished. The final cell also exits
after `keepalive_min`. Stopping invalidates the temporary endpoint and key.
"""


LAUNCH = """## Step 6 — Start Qwen3.8-27B (leave this cell running)

The script builds the pinned TPU runtime, restores the matching XLA cache, finds
the attached weights, starts vLLM across all eight TPU chips, opens a temporary
Cloudflare tunnel, and performs a short self-test. The endpoint may return 502
until the `READY` banner appears.

Run the final cell and watch the output. Expected progress is:

1. `STEP 1/6` — install and verify the pinned TPU runtime.
2. `STEP 2/6` — restore the XLA compile cache.
3. `STEP 3/6` — locate the mounted 55 GB weights.
4. `STEP 4/6` — start vLLM across eight TPU chips.
5. `STEP 5/6` — create the temporary Cloudflare URL.
6. `READY` — copy the endpoint and generated key, then use the Step 5 test.

The cell looking continuously active after `READY` is correct: that active cell
is keeping your API server alive. Stop the Kaggle session when you are done.
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
            markdown("introduction", INTRO),
            markdown("kaggle-setup", SETUP),
            code("check-tpu", TPU_CHECK),
            markdown("input-setup", INPUTS),
            code("check-inputs", INPUT_CHECK),
            markdown("config-help", CONFIG_DOC),
            code("configuration", CONFIG),
            markdown("input-synthesis", SYNTHESIS),
            markdown("server-source", SERVER_NOTE),
            code(
                "embedded-server",
                "%%writefile serve_qwen38.py\n" + server,
                {"jupyter": {"source_hidden": True}, "tags": ["hide-input"]},
            ),
            markdown("client-guide", CLIENT_GUIDE),
            markdown("troubleshooting", TROUBLESHOOTING),
            markdown("launch-help", LAUNCH),
            code("launch-server", "!python serve_qwen38.py\n"),
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
    NOTEBOOK.write_text(expected)
    print(f"wrote {NOTEBOOK.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
