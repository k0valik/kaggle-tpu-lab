# Ornith-1.5-35B-A3B (CRACK) — Kaggle TPU v5e-8 / P100 dual-path (llama.cpp / vllm-tpu)

Serve `Ornith-1.5-35B-A3B` (CRACK-abliterated MoE) through an OpenAI-compatible API.
One script, two hardware paths — the kernel checks what it was given:

1. **TPU v5e-8** (`/dev/apex_*` present) → bf16 safetensors
   (`huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated`, ~70 GB) via vllm-tpu 0.28.0,
   TP=8, 262k ctx, text-only. **Experimental**: `qwen3_5_moe` support in vllm-tpu
   0.28.0 is unverified — if the engine rejects the arch, the log says so explicitly.
   No compile-cache dataset for this model: cold start ~35-50 min.
2. **GPU P100** (`nvidia-smi`) → llama.cpp, **Q4_K** (21.7 GB). Q4_K > 16 GB VRAM, so
   routed experts are split with `--n-cpu-moe`: the ladder tries 20 → 30 → 40 layers'
   experts on CPU (VRAM ~14.7 → 10.5 → 6 GB), then a 131072-ctx fallback. First healthy
   config wins. Decode ~25-50 tok/s (experts partially CPU-resident, mmap'd from SSD).
3. **Neither** → clear failure message (the known Kaggle silent-CPU-fallback case).

## Files

- `ornith-1.5-35b-a3b-p100.ipynb` — the manual P100 notebook (Q4_K ladder).
- `serve_ornith.py` — the dual-path script kernel (same file for both accelerators).
- `kernel-metadata.json` — push as **GPU P100**: `kaggle kernels push -p ornith/`
- `kernel-metadata-tpu.json` — push as **TPU**: copy over `kernel-metadata.json`,
  edit the slug to taste, push (`enable_tpu: true`).

Progress publishes to ntfy topic `ktl-ornith-3f9c2b7e51a04d68` (both paths).

## What to expect

| | P100 — Q4_K split | TPU v5e-8 — bf16 (if supported) |
|---|---|---|
| Weights | GGUF Q4_K 21.7 GB (experts CPU/GPU split) | bf16 safetensors ~70 GB of 128 GB HBM |
| Decode | ~25-50 tok/s (est.) | ~150-300 tok/s (est.) |
| Context | 262144 (q4_0 KV ≈ 1.3 GB) | 262144 |
| Time to READY | ~8-10 min (bigger download) | ~35-50 min (cold compile) |
| Risk | low | medium (arch support unverified) |

## Wiring into opencode

Same key and alias on both paths (`Ornith-1.5-35B-A3B-CRACK`):

```jsonc
"baseURL": "https://<xxx>.trycloudflare.com/v1",
"apiKey":  "1e0afcc97b0ba77076ef35a63664d578",
"model":   "Ornith-1.5-35B-A3B-CRACK"
```
