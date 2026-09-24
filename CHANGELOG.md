# Changelog — k0valik/kaggle-tpu-lab

Repo-level changelog from the fork point (`1aa1f08`, upstream
`ARahim3/kaggle-tpu-lab` HEAD at fork time). Per-commit detail lives in
`git log`; only substantial changes are recorded here. (The Tauri
companion keeps its own `CHANGELOG-0.4.5.txt`.)

## Unreleased

- Stage B (next): generic Qwen3.8-27B weights source (see `plans/B-WEIGHTS-SOURCE.md`).

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
