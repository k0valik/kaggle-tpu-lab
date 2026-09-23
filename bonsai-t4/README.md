# Ternary-Bonsai-2-27B-Abliterated (PQ2_0) — Kaggle T4×2 kernel

Uncensored **Qwen3.8-27B** (dense) in PrismML's ternary 2-bit pack: **7.21 GB**,
ggml type 142. Abliterated at the quant-code level (refusal 37/40 → 0/40, medium mode,
capability within measurement noise).

## Files

- `serve_bonsai.py` — the kernel script (push as-is)
- `kernel-metadata.json` — GPU kernel, id `mikiasendale/bonsai-t4-serve`

```bash
cp kernel-metadata.json /tmp/opencode/bonsai/   # or push in place
kaggle kernels push -p <dir with both files>
```

## What it does

1. **Detect GPU** — any CUDA device; dual-GPU (T4×2) gets `-sm layer` layer-split,
   single GPU runs all layers on one card. CPU-only container → clear failure message.
2. **Install PrismML fork** — prebuilt `linux-cuda-12.4` release tarball (zero build),
   falls back to a source build (~15 min) if the binary won't run.
   **Stock llama.cpp cannot run this file** (custom ternary/Hadamard kernels).
3. **Download** `Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf` (7.21 GB).
4. **Launch** with `--chat-template-kwargs '{"reasoning_effort": "medium"}'`
   (required — template default `xhigh` gives empty answers / weaker abliteration),
   ladder: dual/single `-ngl 99` @ 65536 → 32768 ctx → CPU fallback.
5. Tunnel + READY banner + self-test + keepalive, same pattern as the ornith kernel.
   Progress on ntfy topic `ktl-bonsai-6d2f8e1a4b7c3905`.

## Expected speed (T4×2)

| Config | Decode |
|---|---|
| 2× T4 layer split (default) | **~20-30 tok/s** |
| 1× T4 | ~12-20 tok/s |
| CPU fallback | ~2-3 tok/s |

Reference: L4 = 29.8 tok/s TG128 on PQ2_0 per PrismML's table; T4 derated ~30-50%.
7.21 GB on 16 GB cards leaves lots of room; 65536 ctx is conservative on purpose.

## Wiring into opencode

```jsonc
"baseURL": "https://<xxx>.trycloudflare.com/v1",
"apiKey":  "1e0afcc97b0ba77076ef35a63664d578",
"model":   "Ternary-Bonsai-2-27B-Abliterated"
```

Send `reasoning_effort: "medium"` in requests (or rely on the server's
`--chat-template-kwargs` default set at launch).

## Session notes

- 9 h cap, ~20 h/week GPU quota; new Cloudflare URL each session → rewire baseURL.
- Stop kernels in the Kaggle UI (no stop API). Stopped kernels may vanish → re-push.
