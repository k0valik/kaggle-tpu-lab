# OrcaRouter Qwen3.8-27B Uncensored FP8 on Kaggle TPU v5e-8

This repo now includes an **FP8-specific Kaggle path** for:

`orcarouter/Qwen3.8-27B-Uncensored-FP8`

The original BF16 files remain intact. Use the new FP8 launcher/notebook when you want the OrcaRouter checkpoint.

## What changed

- Uses the OrcaRouter **Safetensors block-FP8** checkpoint (~30.9 GB).
- Keeps Kaggle's TPU **v5e-8** and `TP=8`.
- Lets vLLM read the checkpoint's `quantization_config` directly. It intentionally does **not** pass `--quantization fp8`.
- Preserves the existing Qwen reasoning parser, tool-calling parser, MTP speculative decoding patch, OpenAI-compatible endpoint, Anthropic-compatible endpoint (when provided by the installed vLLM), and Cloudflare tunnel.
- Does **not** reuse the repo's BF16 XLA cache by default. FP8 changes the compiled graph, so the BF16 cache should not be treated as a valid warm-start bundle.
- Downloads from Hugging Face unless you supply a Kaggle dataset mirror of the FP8 checkpoint.

## Hugging Face access is required

The model repository is public but gated by access conditions.

1. Open `https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8`.
2. Accept the model's access conditions.
3. Create a Hugging Face **read** token.
4. In Kaggle, create a secret named **`HF_TOKEN`** and grant the notebook/kernel access.

The token is read at runtime. The new launcher does not embed it into the pushed kernel or print it.

## Option A: terminal launcher

Install/authenticate the Kaggle CLI as usual, then:

```bash
python launch_fp8.py serve
```

Useful variants:

```bash
# Skip the vision tower for a faster text/coding-agent startup
python launch_fp8.py serve --text-only

# Lower context while validating the FP8 path
python launch_fp8.py serve --text-only --max-model-len 131072

# Follow an existing run
python launch_fp8.py status -f

# Terminate the Kaggle TPU session
python launch_fp8.py stop
```

If you created a Kaggle dataset containing the complete FP8 checkpoint:

```bash
python launch_fp8.py serve --weights-dataset yourname/qwen38-uncensored-fp8
```

That avoids downloading ~30.9 GB on every fresh Kaggle session.

## Option B: Kaggle notebook

Upload/open:

`notebook/qwen38-fp8-tpu-serve.ipynb`

Set:

- Accelerator: **TPU VM v5e-8**
- Internet: **ON**
- Secret: **HF_TOKEN**

Then run top to bottom.

## Important XLA-cache note

Do **not** attach `rahim3/qwen38-tpu-env-v5e8` expecting its BF16 graph cache to accelerate this FP8 checkpoint. The runtime wrapper deliberately avoids selecting that bundle by default.

A future FP8-specific cache can be supplied with:

```bash
python launch_fp8.py serve --env-dataset owner/qwen38-fp8-tpu-env-v5e8
```

## v5e / FP8 upstream status

The checkpoint itself declares 128×128 block-FP8 and is designed to be loaded by vLLM from its `config.json`. The current TPU stack's general FP8 hardware-acceleration matrix is not optimized for v5/v6 in the same way as INT8/INT4.

Also note the open `tpu-inference` issue #3399: a Qwen3.5+ weight-placement/resharding bug on single-host v5e-8 can hit a repeatable 4 GiB VFIO mapping failure. The reporter validated that a local direct-sharding correction also allowed the official Qwen3.8-27B-FP8 checkpoint to serve end-to-end. The exact patch has not been published in that issue, so this repo does not invent or silently apply one. If you hit that exact error, use an upstream release/commit that contains the eventual loader fix, or apply the upstream patch once published.

## Files

- `kernel/serve_qwen38_fp8.py` — thin FP8/auth wrapper around the original kernel.
- `launch_fp8.py` — terminal launcher for the FP8 path.
- `notebook/qwen38-fp8-tpu-serve.ipynb` — Kaggle UI notebook.
- `kernel/serve_qwen38.py` — original serving engine, retained unchanged.

## Served model name

The API advertises:

`qwen3.8-27b-uncensored-fp8`

Example:

```bash
curl "$BASE/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-uncensored-fp8",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The upstream model has had its refusal behavior substantially removed. Review the model card and add your own moderation/abuse controls before exposing the endpoint to other users.
