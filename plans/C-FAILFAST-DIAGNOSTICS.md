# Stage C — Fail-fast + honest diagnostics

## Goal

Die in seconds with a clear message on sessions that can never succeed,
and report the true root cause when the server dies — instead of burning
quota on a 20-minute doomed run.

## Context

Fresh/unverified Kaggle accounts often start sessions with no TPU, no
internet, or poisoned `TPU_WORKER_*` env vars (Kaggle metadata WARNING
text breaks PJRT mesh init). The append-mode `vllm.log` also misattributes
rerun failures to old runs. References: PR #5 (Kitkitkittt:
`tpu_topology_ok()` gate), PR #8 (codewith-aditya: `sanitize_tpu_env()`,
`internet_check()`, `_LOG_START` byte-offset, `server_died()` hints).

## Work items

1. Kernel(s): `tpu_topology_ok()` — `jax.devices()` must be 8×TPU; else
   `publish("failed", step=...)` + exit BEFORE runtime install / weights
   load. Place after `install_runtime` only if the probe needs the venv.
   Port to `glm53-flash` only if trivial (secondary).
2. Both kernels: `sanitize_tpu_env()` — pop `TPU_WORKER_HOSTNAMES` /
   `TPU_WORKER_ADDRS` before `tpu_check()`/`preflight()`.
3. Both kernels: `internet_check()` — ~2 s PyPI probe before pip /
   `fetch_cloudflared`; fail-fast with `no-internet` step.
4. Qwen kernel: `_LOG_START` byte-offset — record `RAW_LOG` size in
   `launch_server()`; `server_death_report()` reads only
   `read_bytes()[_LOG_START:]`. Consider same for GLM (currently lacks it).
5. `server_died()` hints: generic "attach `/kaggle/working/vllm.log`"
   fallback + phase-gated flaky-start note; `returncode==0` with no cause
   → external-stop hint (Save & Run All vs interactive session).
6. GLM keepalive: one-line `scheduler exited before keepalive_min` log
   before `publish("stopped")` (if still applicable in current tree).
7. Notebook preflight cells (`check-tpu`, `check-inputs`) + troubleshooting
   table — regenerated from kernel, no fork URLs.
8. Regenerate affected notebook cells.

## PRs consulted (double-pass)

- [#5](https://github.com/k0valik/kaggle-tpu-lab/pull/5) (Kitkitkittt):
  `tpu_topology_ok()` gate + notebook preflight cells + troubleshooting
  table. IGNORE GPU path, fork URLs. Double-pass: re-check
  `download_checked`/`safe_extract` idioms (parked for Stage D) and whether
  the `INPUT_AUDIT.md` synthesis notes hold anything missed.
- [#8](https://github.com/k0valik/kaggle-tpu-lab/pull/8) (codewith-aditya):
  `sanitize_tpu_env()`, `internet_check()`, `_LOG_START` triple,
  `server_died()` hints, scheduler-exit line. IGNORE KYC prose, INT8 essay,
  backslash regression. Double-pass: re-check the `server_death_report`
  hint-split hunk (parked as REVIEW — subsumed by the sanitizer?).

## Non-goals

No KYC prose, no bf16-vs-INT8 essays (PR #8 docs rejected). No GPU
notebook. No Tauri changes.

## Acceptance (semantic; live TPU deferred)

- [ ] `py_compile` on touched kernels; notebooks valid JSON; embedded
      cells match kernel sources.
- [ ] Pure functions covered by local unit tests where extracted:
      env-sanitizer (poisoned vs clean env), `_LOG_START` slicing against
      a fixture log, hint selection for `(returncode, phase, cause)`
      triples.
- [ ] Gating order verified by reading the boot sequence: no large
      download/compile precedes the TPU + internet gates.
- [ ] Deferred-live: CPU-only session fails in seconds; poisoned-env
      session boots; rerun-after-failure attributes the current run.
