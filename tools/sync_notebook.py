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


INTRO = """# Qwen3.8-27B (bf16) on a free Kaggle TPU v5e-8

This notebook serves **Qwen3.8-27B in bf16** with vLLM on Kaggle's free TPU and
creates a temporary, API-key-protected OpenAI-compatible endpoint.

The performance figures shown in this notebook are measurements reported by the
[original kaggle-tpu-lab project](https://github.com/ARahim3/kaggle-tpu-lab), not
new measurements from this fork. The reviewed source for this version is at
[Kitkitkittt/kaggle-tpu-lab](https://github.com/Kitkitkittt/kaggle-tpu-lab).

## Before you run: three settings in the right sidebar

1. Choose **Accelerator -> TPU VM v5e-8**.
2. Turn **Internet on** (the runtime and temporary Cloudflare tunnel need it).
3. Add both public inputs:
   - `rahim3/qwen3-8-27b-bf16` (55 GB bf16 weights)
   - `rahim3/qwen38-tpu-env-v5e8` (matching XLA cache and helper binary)

Then run top to bottom. The last cell remains active because it is the server.
Wait for the `READY` banner before using the displayed endpoint and API key.
If startup reports `TPU topology check FAILED` or only `cpu:0`, stop the session,
reselect **TPU VM v5e-8** in Session options, confirm account verification/quota,
and rerun. The script stops before loading the weights when no TPU was allocated.

> Security: your Kaggle access token never belongs in this notebook. Each run
> generates a fresh inference API key. Treat the endpoint and key as temporary
> secrets, and stop the Kaggle session when finished.
"""


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


CONFIG_DOC = """## Configuration

This published version defaults to a quota-conscious text-serving profile:

- `max_model_len: 262144` keeps the model's native context with four concurrent
  sequences. For throughput, use `131072` with `max_num_seqs: 16`.
- `mtp_tokens: 3` enables the bundled, verified state-rollback patch for MTP
  speculative decoding. Set it to `0` to disable speculative decoding.
- `text_only: true` skips the vision tower and its startup work. Set it to
  `false` if your client needs image inputs.
- `fast_start: true` exposes the endpoint before all request shapes are warmed.
  The first unusual shape can pause while XLA compiles it; later requests reuse it.
- `keepalive_min: 30` auto-stops serving after 30 minutes. Raise it only when
  needed, up to Kaggle's session limit.

The prebuilt cache covers the documented 262k/4 and 131k/16 profiles. Other
combinations can work, but compile cold and take longer to start.
"""


SYNTHESIS = """## Why this is the TPU path

Two reviewed input notebooks run quantized GGUF files through CUDA/llama.cpp on
dual T4 GPUs. Their download and chat ideas are useful GPU fallbacks, but CUDA
binaries, GPU layer offload, and `tensor_split` do not apply to a TPU VM.

This notebook instead follows the current TPU-native implementation: bf16
safetensors, eight-way TPU tensor parallelism, vLLM TPU, a matching XLA compile
cache, generated authentication, and automatic shutdown. The full comparison
and the items deliberately not copied are documented in
[`docs/INPUT_AUDIT.md`](https://github.com/Kitkitkittt/kaggle-tpu-lab/blob/main/docs/INPUT_AUDIT.md).
"""


SERVER_NOTE = """## Serving script

The next cell writes the canonical serving script from this repository. Keeping
the notebook generated from `kernel/serve_qwen38.py` prevents the UI version and
the reviewed source from silently drifting apart.
"""


LAUNCH = """## Launch (leave this cell running)

The script builds the pinned TPU runtime, restores the matching XLA cache, finds
the attached weights, starts vLLM across all eight TPU chips, opens a temporary
Cloudflare tunnel, and performs a short self-test. The endpoint may return 502
until the `READY` banner appears.

The launch also verifies that JAX sees exactly eight TPU devices. A CPU-only
session is a Kaggle provisioning/settings issue, not a model-loading failure.

Use the values printed in that banner:

```bash
curl <ENDPOINT>/chat/completions \\
  -H "Authorization: Bearer <API_KEY>" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "chat_template_kwargs": {"reasoning_effort": "low"}
  }'
```

Stopping the cell or Kaggle session shuts the endpoint down. The generated API
key is unrelated to your Kaggle account token.
"""


def source(text):
    return text.splitlines(keepends=True)


def markdown(cell_id, text):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source(text)}


def code(cell_id, text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
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
            code("configuration", CONFIG),
            markdown("config-help", CONFIG_DOC),
            markdown("input-synthesis", SYNTHESIS),
            markdown("server-source", SERVER_NOTE),
            code("embedded-server", "%%writefile serve_qwen38.py\n" + server),
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
