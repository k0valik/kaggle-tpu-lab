#!/usr/bin/env python3
"""
FP8 entrypoint for Kaggle TPU v5e-8.

This wrapper keeps the original serving engine intact, but switches its model
source to orcarouter/Qwen3.8-27B-Uncensored-FP8, makes Hugging Face gated-model
authentication explicit, and prevents accidental reuse of the BF16 XLA bundle.
"""

import json
import os
import sys
from pathlib import Path

MODEL_ID = "orcarouter/Qwen3.8-27B-Uncensored-FP8"
SERVED_MODEL_NAME = "qwen3.8-27b-uncensored-fp8"

CFG = None  # __LAUNCHER_CONFIG__

HERE = Path(__file__).resolve().parent
BASE_KERNEL = HERE / "serve_qwen38.py"
CONFIG_FILE = HERE / "serve_config.json"

FP8_DEFAULTS = {
    "hf_model_id": MODEL_ID,
    "served_model_name": SERVED_MODEL_NAME,
    # A blank dataset would make the base kernel probe /kaggle/input itself.
    # Use a sentinel instead so it cleanly falls back to Hugging Face.
    "weights_dataset": "__hf_fp8_download__",
    # The shipped rahim3/qwen38-tpu-env-v5e8 cache was compiled for BF16.
    # Never select it implicitly for the FP8 graph.
    "env_dataset": "__fp8_env_not_attached__",
    "vllm_tpu_version": "0.28.0",
}


def _get_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def _load_user_config():
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception as exc:
        raise SystemExit(f"Invalid {CONFIG_FILE.name}: {exc}") from exc


cfg = {**FP8_DEFAULTS, **_load_user_config(), **(CFG or {})}
if not cfg.get("weights_dataset"):
    cfg["weights_dataset"] = "__hf_fp8_download__"
if not cfg.get("env_dataset"):
    cfg["env_dataset"] = "__fp8_env_not_attached__"

# A Hugging Face token is required only when the checkpoint must be downloaded.
# A complete Kaggle dataset mirror can run without exposing any HF credential.
token = _get_hf_token()
needs_hf_download = cfg["weights_dataset"] == "__hf_fp8_download__"
if needs_hf_download and not token:
    raise SystemExit(
        "\nHF_TOKEN is required because orcarouter/Qwen3.8-27B-Uncensored-FP8 "
        "requires accepting its Hugging Face access conditions.\n"
        "1. Open the model page and accept the conditions.\n"
        "2. Create a Hugging Face read token.\n"
        "3. Add it to Kaggle Secrets with the name HF_TOKEN and grant this notebook/kernel access.\n"
        "Alternatively attach a Kaggle dataset containing the complete checkpoint and set "
        "weights_dataset in serve_config.json.\n"
    )
if token:
    os.environ["HF_TOKEN"] = token

# Never leak credentials into serve_config.json.
cfg.pop("hf_token", None)
CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

if not BASE_KERNEL.exists():
    raise SystemExit(
        f"Missing {BASE_KERNEL}. The FP8 wrapper must be distributed beside "
        "kernel/serve_qwen38.py."
    )

# Run the original kernel after a few targeted, runtime-only edits. Keeping the
# large embedded MTP rollback patch in one source file avoids patch drift.
src = BASE_KERNEL.read_text()
src = src.replace(
    "Serve Qwen3.8-27B (bf16) on a Kaggle TPU v5e-8 with vLLM.",
    "Serve Qwen3.8-27B Uncensored FP8 on a Kaggle TPU v5e-8 with vLLM.",
)
src = src.replace(
    'banner(3, "Model weights", "55 GB bf16 safetensors")',
    'banner(3, "Model weights", "30.9 GB block-FP8 safetensors")',
)
src = src.replace(
    "loading 55 GB of weights",
    "loading ~31 GB of FP8/BF16 mixed weights",
)
src = src.replace(
    '"qwen38-tpu-env*"',
    '"qwen38-fp8-tpu-env*"',
)
src = src.replace(
    'model_path = snapshot_download(CFG["hf_model_id"], allow_patterns=[',
    'model_path = snapshot_download(CFG["hf_model_id"], token=os.environ.get("HF_TOKEN"), allow_patterns=[',
)
src = src.replace(
    '"--served-model-name", cfg["served_model_name"],\n'
    '            "--reasoning-parser", "qwen3"]',
    '"--served-model-name", cfg["served_model_name"],\n'
    '            "--trust-remote-code",\n'
    '            "--reasoning-parser", "qwen3"]',
)

# Execute as the serving script. config.json in the model checkpoint declares
# the block-FP8 quantization; intentionally do NOT add --quantization fp8.
os.chdir(HERE)
exec(compile(src, str(BASE_KERNEL), "exec"), globals(), globals())
