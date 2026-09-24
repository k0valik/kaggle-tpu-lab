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

## PR reference map (double-pass checklist)

All open PRs on `k0valik/kaggle-tpu-lab` — semantic references only, never
merge candidates. Raw diffs archived at `/tmp/opencode/pr-diffs/pr-N.diff`
(local only; re-fetch with `gh pr diff N` if missing). After all stages
land, do a second pass over each PR against this table to catch misses.

| PR | Author | Title | Verdict summary | Stage(s) |
|---|---|---|---|---|
| [#1](https://github.com/k0valik/kaggle-tpu-lab/pull/1) | phakoda | OrcaRouter FP8 path | IGNORE feature; LIFT token-plumbing idea, runtime-secret hygiene, FP8-cache rule, trust-remote-code question | A, B |
| [#2](https://github.com/k0valik/kaggle-tpu-lab/pull/2) | chynggi | Serve finetunes, not just base | LIFT `*.jinja`, token, `chat_template()` fallback, `hf_token()` chain, empty-dataset filter; IGNORE presets/wrappers | B |
| [#3](https://github.com/k0valik/kaggle-tpu-lab/pull/3) | xiaotian1171 | Huihui abliterated + weights builder | IGNORE defaults/docs; REVIEW generic `find_input()` fallbacks (landed), parameterized `build-weights` (split later) | B |
| [#4](https://github.com/k0valik/kaggle-tpu-lab/pull/4) | tonyrishwain | Notebook template/weights + tunnel fix | LIFT `id_no` pop + returncode push check; claimed tunnel fix has no hunk | E |
| [#5](https://github.com/k0valik/kaggle-tpu-lab/pull/5) | Kitkitkittt | Guides + fail-fast | LIFT `tpu_topology_ok()`, state `0600`, CI, sync `--check`, preflight cells; IGNORE GPU path | C, E (+D idioms) |
| [#6](https://github.com/k0valik/kaggle-tpu-lab/pull/6) | qdubois | Secrets out, tunnel downloads | LIFT pinned-cloudflared logic, `--host 127.0.0.1`, redact, atomic `0600`; IGNORE env-dataset deletion | D |
| [#7](https://github.com/k0valik/kaggle-tpu-lab/pull/7) | nxhung1610 | DFlash2 TPU | IGNORE entirely (new drafter); keep only the abstract mutual-exclusion rule | none |
| [#8](https://github.com/k0valik/kaggle-tpu-lab/pull/8) | codewith-aditya | Fix kaggle issues | LIFT `sanitize_tpu_env()`, `internet_check()`, `_LOG_START`, `server_died()` hints; IGNORE KYC prose, backslash regression | C |
| [#9](https://github.com/k0valik/kaggle-tpu-lab/pull/9) | KiVixx | Stable custom endpoint | LIFT `optional_dataset()`, HF preflight/validation, XET toggle, secrets plumbing, named tunnel, sync concept; IGNORE uncensored notebooks, zh-TW doc, queue monitor | B, D, E |

## Double-pass (2026-09-24, all stages landed)

Six sequential verifications (V1–V6), then one fix batch. Accepted gaps,
all landed: config-compat check + MTP-head fallback (V1), 2 trivial GLM
ports (V2), XLA tar-slip guard + fail-hard/push-secrecy tests (V3),
BUILD_CONFIGS sync comment + pin test (V4), ARahim3→k0valik URL hygiene
(V5), trust-remote-code comment + MTP-on-FP8 warning + #3399 tracking
(V6). Confirmed rejections: `--use-xet` flag, HMAC protocol, SESSION_DIR,
template-flow helpers, hint-split, GPU paths, DFlash2, all hardcoded
checkpoints. No MISSED hunk in any PR (V2/V4/V6 censuses). Standing
deferred-live items: TPU re-measurement, MTP A/B (bf16 + FP8), APC
hit-rate, JSON-mode crash repro, trust-remote-code probe, named-tunnel
e2e, CPU-landing of build-weights.
