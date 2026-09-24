# Stage B — Generic Qwen3.8-27B weights source (incl. own FP8)

## Goal

Serve any compatible Qwen3.8-27B checkpoint — prebaked BF16 dataset
(default), an uploaded Kaggle dataset (e.g. the owner's own FP8 quant), or
a HuggingFace download (`--hf-model-id`) — without hardcoding any
third-party checkpoint.

## Context

- Current: `WEIGHTS_DATASET="rahim3/qwen3-8-27b-bf16"` hardcoded in
  `launch.py`; kernel falls back to HF `snapshot_download` of 55 GB to
  `/tmp` (likely fatal on Kaggle disk) with no token support and
  `allow_patterns` missing `*.jinja`.
- Already in baseline (Haz merge): `complete_bf16_repo()` validation +
  `*.jinja` in download patterns. Keep and extend.
- Reference ideas (lift concepts only): PR #2 (chynggi: `hf_token()`,
  `chat_template()` fallback, empty-dataset filter — best design), PR #9
  (KiVixx: `optional_dataset()`, `HfApi` size preflight, progress events,
  `validate_model_snapshot`, `HF_HUB_DISABLE_XET`), PR #3 (xiaotian1171:
  parameterized `build-weights` CPU-mirror flow, generic `find_input()`
  fallbacks), PR #1 (phakoda: runtime-secret hygiene, FP8-must-not-reuse-
  BF16-cache rule).

## Work items

1. `launch.py`: add `--hf-model-id` (default `Qwen/Qwen3.8-27B`),
   `--served-model-name`, `--weights-dataset` accepting `""`/`none` to
   mean "HF download inside kernel"; `dataset_sources` filters empties;
   thread `hf_model_id/served_model_name` through CFG. CLI-first; do NOT
   extend the Tauri app's `serve_args()` (contract freeze).
2. Kernel: `hf_token()` lookup chain (launcher CFG → `HF_TOKEN` env →
   Kaggle secret at runtime; never embed raw token in pushed source,
   never log it); `snapshot_download(..., token=... or None,
   allow_patterns=[..., "*.jinja"])`.
3. Kernel: `chat_template()` helper — `tokenizer_config.json` first,
   `chat_template.jinja` fallback (fixes transformers-v5 `KeyError`).
4. Kernel: weights resolution order — explicit dataset mount (validated by
   `complete_bf16_repo()` + generalized snapshot check) → else HF
   download with `HfApi` size-vs-`shutil.disk_usage(/tmp)` preflight,
   `weights-progress` events, `HF_HUB_DISABLE_XET=1` default +
   `HF_HUB_DOWNLOAD_TIMEOUT`; fail with disk-stats in the error.
5. FP8 rule: when `hf_model_id`/dataset differs from the BF16 default,
   the BF16 XLA cache must NOT be treated as a hit (different graphs);
   log this explicitly.
6. Generic `find_input()` fallbacks for UI-vs-API mount path variants
   (no model-specific globs).
7. Optional: parameterized `build-weights` launcher subcommand + notebook
   (CPU `snapshot_download` for later dataset creation), with token +
   `*.jinja` support. Skip if it bloats the stage; can split to B2.
8. Regenerate the Qwen notebook cell from the edited kernel.

## B2 — free-CPU-kernel weights builder (ADOPTED from PR #3)

The trick: TPU quota (~20 h/week) is the scarce resource; CPU kernels are
separate/free. Instead of downloading a 55 GB checkpoint on the TPU run
(burning TPU hours on download + cold compile), push a small CPU kernel
that `snapshot_download`s the checkpoint to `/kaggle/working/weights`,
then create a Kaggle dataset from its output in the UI and attach that
dataset to the TPU run. (If the owner holds the quant locally, they can
create the dataset from local files directly — the CPU trick is for
HF-hosted checkpoints, incl. gated ones.)

Implement CLI-first: `launch.py build-weights --hf-model-id X --slug Y`
(pushes a CPU script kernel templated with the repo ID + `*.jinja`
patterns + token support, waits, prints "create dataset from output"
instructions). No TPU flag, no new dependencies. Notebook equivalent
optional — owner works from the terminal. Docs land in Stage F.

## PRs consulted (double-pass)

- [#2](https://github.com/k0valik/kaggle-tpu-lab/pull/2) (chynggi, finetunes):
  LIFTED `*.jinja` + token in `snapshot_download`, `chat_template()`
  fallback, `hf_token()` chain, empty-dataset filter. IGNORED presets,
  `quickstart.sh/ps1`, `--served-model-name` docs angle (flag itself taken).
  Double-pass: re-check the config-compat pre-flight script idea (REVIEW).
- [#1](https://github.com/k0valik/kaggle-tpu-lab/pull/1) (phakoda, FP8):
  LIFTED runtime-secret hygiene (preferred over #2's `--hf-token` embed)
  and the FP8-must-not-reuse-BF16-cache rule (landed + hardened to a fresh
  empty dir). IGNORED `launch_fp8.py` wrapper, sentinels, FP8 docs.
- [#3](https://github.com/k0valik/kaggle-tpu-lab/pull/3) (xiaotian1171):
  LIFTED generic `find_input()` fallbacks. IGNORED Huihui defaults/docs.
  ADOPTED as B2: the free-CPU-kernel weights-builder trick (see below).
- [#9](https://github.com/k0valik/kaggle-tpu-lab/pull/9) (KiVixx):
  LIFTED `optional_dataset()`, `HfApi` preflight, `HF_HUB_DISABLE_XET`,
  validation + progress (log-lines-only chosen over the progress thread).
  Secrets/named-tunnel parts belong to Stage D, not B — double-pass must
  confirm nothing weights-related was left (e.g. `--use-xet` flag is D).

## Non-goals

No third-party checkpoint presets, no `quickstart.sh/ps1`, no
`launch_fp8.py` `exec()`-wrapper, no uncensored/abliterated docs (PRs
#1/#3 docs rejected). No Tauri changes. No vLLM version work (Stage A).

## Acceptance (semantic; live TPU deferred)

- [ ] `py_compile` on touched files; notebook JSON valid; embedded kernel
      cell matches kernel source.
- [ ] New pure logic unit-tested locally if extracted as functions:
      `optional_dataset` mapping, `chat_template` fallback against
      fixtures (v4-style `tokenizer_config.json` vs v5-style
      `chat_template.jinja`), snapshot-validation against fixtures.
      (If kept inline in kernel flow, note as untestable-inline instead.)
- [ ] No hardcoded third-party model IDs/datasets in code or docs (grep
      for `orcarouter|huihui|Serenity|JonathanColetti|king888` → clean).
- [ ] Secrets: grep shows no `api_key`/token in `publish()` payloads or
      new log lines.
- [ ] Deferred-live: BF16 cache-hit run, HF-download run, FP8-dataset run,
      gated-repo run, MTP on FP8.
