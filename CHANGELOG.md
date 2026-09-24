# Changelog — k0valik/kaggle-tpu-lab

Repo-level changelog from the fork point (`1aa1f08`, upstream
`ARahim3/kaggle-tpu-lab` HEAD at fork time). Per-commit detail lives in
`git log`; only substantial changes are recorded here. (The Tauri
companion keeps its own `CHANGELOG-0.4.5.txt`.)

## Unreleased

- Live TPU validation (owner): re-measurement, MTP A/B (bf16 + FP8), APC
  hit-rate, JSON-mode crash repro, trust-remote-code probe, named-tunnel
  e2e, build-weights CPU landing.

## 2026-09-24 — `f62b1da` — Double-pass over all 9 PRs

- Six sequential verifications (V1–V6), then one fix batch. Result: no
  MISSED hunk in any PR — every idea is landed, parked with cause, or
  rejected with cause; PRs are now closable with pointers.
- Landed: config-compat cache reuse, MTP-head auto-fallback, XLA
  tar-slip guard, 2 GLM ports, BUILD_CONFIGS pin + test, URL hygiene,
  trust-remote-code/MTP-on-FP8/#3399 tracking. 54/54 tests green.

## 2026-09-24 — `308396e` — Stage F: docs refresh

- Both READMEs brought onto the new baseline: stale 0.28.0-era numbers
  explicitly marked (not silently kept), version references at 0.29.0,
  new docs for custom checkpoints, `build-weights`, fail-fast gates,
  named tunnel + stable key, and the sync/CI workflow. Every stale
  claim swept and dispositioned.

## 2026-09-24 — `0d92566` — Stage E: launcher hardening + sync + CI

- Push handling hardened at all three sites (fail on returncode, note
  on output drift); metadata audit proved no `id_no` footgun exists.
- `tools/sync_notebook.py` ends hand-sync drift: content-located cells,
  both recipes, `--check` mode (GLM byte-identical to old generator).
- Minimal CI (compile + 38 tests + sync check + JSON, no publishing,
  minutes-cheap) and a contract test pinning the frozen Tauri surface.

## 2026-09-24 — `bcec5be` — Stage B2: free-CPU-kernel weights builder

- New `launch.py build-weights --hf-model-id`: mirrors any HF checkpoint
  via a free CPU kernel (no TPU quota burned), then guides dataset
  creation for the TPU run. Adopted from PR #3's trick, generalized:
  no hardcoded checkpoint, `*.jinja` included, runtime-only `HF_TOKEN`.
- Deliberately stateless: never clobbers the TPU serve state that
  `status`/`stop` depend on.

## 2026-09-24 — `63751fd` — Stage D: tunnel + supply-chain hardening

- `cloudflared` 2026.9.1 pinned + independently re-verified sha256,
  atomic install, pre-exec re-hash, foreground fail-hard (no more
  unpinned `latest` binary or unhashed dataset copy).
- vLLM binds `127.0.0.1`; secrets scrubbed from logs/ntfy (ready event
  carries no key); atomic `0600` launcher state.
- Opt-in stable named tunnel + stable API key; quick-tunnel default kept.
- Implement → independent-review → land pipeline: reviewer re-derived
  the digest and verified the Tauri contract; 32/32 tests green.

## 2026-09-24 — `0e443be` — Stage C: fail-fast gates + honest diagnostics

- Kernel dies in seconds (not after 20 doomed minutes) on CPU-only,
  poisoned-env, or offline sessions: env sanitize → topology gate →
  internet check, all before venv build / downloads.
- Append-mode `vllm.log` no longer misattributes old runs (`_LOG_START`
  offset); `server_died()` distinguishes startup crash vs external
  stop with actionable hints.
- Notebook gains TPU + dataset/scratch preflight cells and a
  troubleshooting table; `tests/test_failfast.py` (7 tests) green.

## 2026-09-24 — `f4c7b56` — Stage B: generic Qwen3.8-27B weights source

- Prebaked BF16 stays default; any compatible checkpoint now servable via
  uploaded Kaggle dataset or `--hf-model-id` download (`--weights-dataset
  none` = HF download). No third-party presets hardcoded.
- Kernel: runtime `hf_token()` chain (never logged), `*.jinja` downloads,
  `chat_template.jinja` fallback, HF size preflight + snapshot validation.
- Non-default weights bypass the BF16 XLA cache via a fresh empty dir
  (stale entries never consulted — correctness, not just messaging).
- `tests/test_weights_source.py`: 12 unit tests green; notebook
  regenerated; semantic-only validation, live runs deferred.

## 2026-09-24 — `140ab53` — Stage A: re-baseline onto `vllm-tpu==0.29.0`

- Bumped the Qwen runtime pin `0.28.0` → `0.29.0` (plugin; core `0.30.0`
  exists but the TPU plugin is still 0.29.0, which is also what core
  0.30.0 pins for TPU — nothing missed).
- Re-ported the MTP rollback patch (`mtp-rollback-v0290.diff`, 7 files,
  dry-run clean on pristine 0.29.0) since upstream `tpu-inference#3178`
  is still open and 0.29.0 rewrote the GDN files; fail-closed kept, MTP
  default stays 3.
- Added `--enable-prefix-caching` (upstream #3422 fixed hybrid-GDN APC;
  platform-guarded under spec-decode). Kept the `--no-async-scheduling`
  workaround (no upstream fix for `__delitem__`).
- Notebook cell regenerated; semantic-only validation (no Kaggle yet),
  live TPU checks deferred and listed in the commit message.

## 2026-09-24 — `6e54602` — Staged consolidation plans + validation policy

- Added `plans/00-OVERVIEW.md` and `plans/A-VLLM-BASELINE.md` … `plans/F-DOCS-REFRESH.md`: self-contained stage files for subagent fan-out.
- Validation is semantic/local only until the owner connects Kaggle (no
  live TPU runs yet); each stage marks acceptance as semantic vs
  deferred-live.

## 2026-09-24 — `9b45af4` — `AGENTS.md`: baseline, frozen Tauri, focus

- Recorded the Haz4rdovisk merge as the consolidation baseline.
- Tauri companion app (`src/`, `src-tauri/`, `scripts/`, packaging files)
  declared vendored/frozen, with the launcher/state/ntfy/probe contract
  that recipe work must not break.
- Scope: Qwen3.8-27B family primary incl. generic configurable weights
  source (own FP8 via Kaggle dataset or `--hf-model-id`); `glm53-flash`
  deferred; latest vLLM baseline.

## 2026-09-24 — `af6973a` — Merge Haz4rdovisk companion, accepted wholesale

- Merged `Haz4rdovisk/kaggle-tpu-lab@6c0ef49` with `--no-ff`
  (conflict-free; our HEAD was its ancestor). Tag: `baseline+haz-companion`.
- Added the Tauri tray companion app (~21k lines: React + Rust, shells out
  to `launch.py`, polls ntfy, probes `GET {endpoint}/v1/models`).
  Security-audited: clean (no exfil/obfuscation/bundled binaries).
- 4 recipe hunks came with it: `machine_shape:TpuV5E8` + `--accelerator`
  on push, `returncode`-checked push, state keys
  `model/submitted_at/keepalive_min`, `complete_bf16_repo()` weights
  validation + `*.jinja` in `snapshot_download` patterns.
