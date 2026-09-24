# Consolidation plan — overview

Baseline: `af6973a` (Haz4rdovisk companion merged wholesale) + `9b45af4`
(`AGENTS.md`). Tag: `baseline+haz-companion`. See `AGENTS.md` for the full
operating rules; this file is the stage index.

## Constraints (repeated so every stage file stays self-contained)

- Open PRs (#1–#9 on `k0valik`) are **semantic references only, never merge
  candidates**. Lift generic fixes hunk-by-hunk; discard fork packaging.
  Cite `Ref: #N` in commits.
- English only. Qwen3.8-27B family primary; `glm53-flash` deferred (port
  generic infra only when free). No new architectures/drafters (DFlash2),
  no hardcoded third-party checkpoints — but **generic configurable
  Qwen3.8-27B weights source is wanted** (prebaked BF16 default; own FP8 or
  compatible checkpoint via Kaggle dataset or `--hf-model-id`).
- Latest `vllm-tpu` is the baseline; `0.28.0` pins/claims are stale.
- Tauri app (`src/`, `src-tauri/`, `scripts/`, packaging files) is frozen.
- Owner has **no Kaggle access**: validation is semantic/local only (see
  `AGENTS.md` validation policy). Every stage lists what was checked
  locally vs deferred-live.

## Key paths

- `launch.py` — CLI: serve/status/stop/build-env, pushes kernel, watches ntfy.
- `qwen38-27b/kernel/serve_qwen38.py` — **source of truth** for Qwen.
- `qwen38-27b/notebook/qwen38-tpu-serve.ipynb` — generated copy; never hand-diverge.
- `qwen38-27b/patches/*.diff` — patch source; `MTP_PATCH_B64` derived via `qwen38-27b/tools/embed_patch.py`.
- `glm53-flash/` — secondary; kernel + JAX engine + generated notebook via `tools/pack_notebook.py`.

## Stages (run in order; each has its own file)

| Stage | File | Goal |
|---|---|---|
| A | `A-VLLM-BASELINE.md` | Re-baseline onto latest `vllm-tpu`; re-validate MTP patch, prefix cache, `__delitem__`, `--trust-remote-code` |
| B | `B-WEIGHTS-SOURCE.md` | Generic Qwen3.8-27B weights: Kaggle dataset OR HF download (`--hf-model-id`), token plumbing, `*.jinja`, snapshot validation |
| C | `C-FAILFAST-DIAGNOSTICS.md` | Fail fast on bad sessions (TPU topology, poisoned env, no internet); honest error reports |
| D | `D-TUNNEL-HARDENING.md` | Pinned+verified `cloudflared`, `--host 127.0.0.1`, secret hygiene, named-tunnel option |
| E | `E-LAUNCHER-SYNC-CI.md` | Launcher robustness, kernel→notebook single-source sync, CI |
| F | `F-DOCS-REFRESH.md` | Rewrite stale READMEs for latest vLLM + new flags |

## Subagent protocol

- A build subagent receives: its stage file + repo checkout. Nothing else
  is assumed known.
- Subagent returns: files changed, and for each acceptance item PASS/FAIL
  with the exact command run.
- Orchestrator validates output against the stage acceptance list, then
  commits (one commit per stage, `Ref: #N` attributions).
