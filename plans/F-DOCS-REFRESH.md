# Stage F — Docs refresh

## Goal

Make the READMEs true again on the new baseline: latest vLLM, generic
weights source, new flags, kept tradeoffs. Last stage so it documents
what was actually built.

## Context

`qwen38-27b/README.md` (250 lines: 130 tok/s table, 0.28.0 GDN story,
prefix-cache-off note, `__delitem__` note, startup-time breakdown) and
root `README.md` predate Stages A–E. Depends on all prior stages — do not
start early.

## Work items

1. `qwen38-27b/README.md`: new vLLM version + measured-or-marked-stale
   numbers (every perf number is either re-measured live or explicitly
   labeled `stale (0.28.0-era, re-measure on TPU)` — never silently kept).
   Update: MTP status, prefix-cache status, JSON-mode/`__delitem__`
   status, `--trust-remote-code` if added, cloudflared pin, cache-match
   rules incl. the FP8-cache-miss rule.
2. Document the generic weights source: `--weights-dataset` (explicit /
   `none` = HF download), `--hf-model-id`, `--served-model-name`, token
   via env/secret, `build-weights` flow if built. No third-party
   checkpoint endorsements.
3. Document fail-fast behavior (Stage C), tunnel options incl. named
   tunnel (Stage D), and the kernel↔notebook sync + CI contract (Stage E).
4. Root `README.md`: model table + launcher usage updated; keep the
   companion section accurate without expanding it (frozen app).
5. Delete or version-stamp every 0.28.0-era claim (grep `0.28.0`,
   `tpu-inference#3178`, `3178` and disposition each hit).

## Non-goals

No new benchmarks invented locally (numbers come from live runs or are
marked stale). No localization. No Tauri docs expansion.

## Acceptance (semantic)

- [ ] Zero un-stamped stale claims (grep `0\.28\.0|3178|130 tok/s|540
      tok/s|900 tok/s|10,300` → each hit either updated or explicitly
      marked stale with version).
- [ ] Every new flag in `launch.py serve --help` appears in the docs.
- [ ] English only; no fork-specific URLs except `k0valik` + upstream
      attributions.
- [ ] Deferred-live: all perf numbers re-measured on TPU by the owner;
      track as a checklist in the commit message.
