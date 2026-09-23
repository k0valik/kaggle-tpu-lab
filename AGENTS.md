# AGENTS.md — k0valik/kaggle-tpu-lab (fork consolidation)

Fork of `ARahim3/kaggle-tpu-lab` (upstream unmaintained). We consolidate in this fork (`k0valik`).
Baseline since `af6973a`: Haz4rdovisk companion merged wholesale
(`Haz4rdovisk/kaggle-tpu-lab@6c0ef49`, tag `baseline+haz-companion`).
That merge added the Tauri tray app + 4 recipe hunks
(`machine_shape:TpuV5E8` + `--accelerator`, `returncode`-checked push,
state `model/submitted_at/keepalive_min`, `complete_bf16_repo` + `*.jinja`).
All consolidation work happens on top of that baseline.

## Tauri companion app (vendored, frozen)

- `src/`, `src-tauri/`, `scripts/`, `package*.json`, `vite.config.ts`,
  `tsconfig.json`, `index.html`, `CHANGELOG-0.4.5.txt` are **accepted
  wholesale and frozen**. No feature work, no refactors.
- Touch only for: (a) a Haz4rdovisk upstream bugfix worth lifting (same
  semantic-reference rule as PRs), or (b) a bug the owner personally hits.
- Contract the app depends on — do not break:
  `launch.py serve/stop` argv (`--model/--keepalive-min/--max-model-len/--mtp/--reasoning-effort/--fast-start/--text-only/--no-async-scheduling/--max-len/--streams`),
  state-file keys (`kernel/topic/api_key/model/submitted_at/keepalive_min`),
  ntfy `publish(phase, ...)` events, `GET {endpoint}/v1/models` probe.
  New launcher flags (e.g. `--hf-model-id`) are CLI-first; the app catches
  up only if needed.

## How to treat open PRs

- Every open PR from another user's fork is a **semantic reference only, never a merge candidate**.
- Each PR was likely based on an older tree. Do a **semantic, hunk-by-hunk review**: lift the generic fix, discard the fork-specific packaging.
- Never `gh pr merge` / `git merge` a consolidation PR. Cherry-pick ideas, rewrite them against the current layout (`qwen38-27b/kernel/serve_qwen38.py`, `glm53-flash/kernel/serve_glm53.py`, top-level `launch.py`).
- Attribute the idea in the commit message (`Ref: #N`), do not preserve the PR's diff verbatim.

## Scope (hard constraints)

1. **English only.** Ignore localization/docs in other languages. If a non-English PR contains a real code fix, lift the code, drop the prose.
2. **Models: Qwen3.8-27B family only (primary).** `glm53-flash` is secondary and out of immediate scope — port generic infra to it only when free.
   - No new architectures, no new drafters (e.g. DFlash2), no hardcoded third-party checkpoints (uncensored/abliterated/FP8 presets) as features.
   - Exception the owner explicitly wants: **generic configurable Qwen3.8-27B weights source** — the prebaked BF16 Kaggle dataset stays default, but the user can serve their **own FP8 quant or any compatible checkpoint** either (a) as an uploaded Kaggle dataset or (b) as an `--hf-model-id` HuggingFace download. Generalize, never hardcode a third-party checkpoint.
3. **Latest vLLM release is the baseline.** `qwen38-27b/README.md` numbers and the `vllm-tpu==0.28.0` pin are stale (PyPI shows a Sep-2026 `vllm-tpu` release). Re-validate on lift: MTP rollback patch (`patches/mtp-rollback-v0280.diff`, port of `tpu-inference#3178`), prefix-caching enablement, `--no-async-scheduling` / `__delitem__` JSON-mode bug, `--trust-remote-code` need, cloudflared pin digest.

## Source of truth

- `qwen38-27b/kernel/serve_qwen38.py` is source of truth for Qwen. Notebook `qwen38-27b/notebook/qwen38-tpu-serve.ipynb` is a **generated copy** — edit the kernel, regenerate, never hand-diverge.
- `patches/*.diff` is the patch source; `MTP_PATCH_B64` in the kernel is derived via `qwen38-27b/tools/embed_patch.py`.
- `launch.py` injected kernel (CFG + ENGINE_B64 substitution on `CFG = None  # __LAUNCHER_CONFIG__`) is ephemeral.
- Secrets never touch disk-plaintext, CLI args, logs, or ntfy: Kaggle secret / env at runtime, `chmod 0600` state, redact before `publish()`.

## Secrets / datasets / tunnel policy

- Weights resolution order: explicit `--weights-dataset` Kaggle mount → else HF `snapshot_download` (with token, `*.jinja` patterns, `HF_HUB_DISABLE_XET=1` default, size preflight, snapshot validation). Empty/`none` dataset must mean "HF download", never a bogus source.
- `cloudflared`: pinned version + sha256 verify + pre-exec re-hash, foreground fetch with fail-fast message. No `latest` unpinned binary, no dataset-bundled binary without hash check. `--host 127.0.0.1` on vLLM; tunnel stays the public entry.
- Prefer stable named-tunnel + stable API-key-from-secret when configured; keep random quick-tunnel as default.
