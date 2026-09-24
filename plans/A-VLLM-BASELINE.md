# Stage A — Re-baseline onto latest vLLM

## Goal

Replace the stale `vllm-tpu==0.28.0` baseline with the latest release and
re-validate every 0.28.0-era assumption before later stages build on them.

## Version target (verified 2026-09-24 via PyPI)

- Core `vllm` latest: **0.30.0**. TPU plugin `vllm-tpu` latest: **0.29.0**
  (the kernel pins the plugin, which pulls a compatible core — do NOT pin
  core 0.30.0 directly; confirm which core 0.29.0-of-the-plugin requires).
- Target: `vllm-tpu==0.29.0` unless research shows a newer plugin release.
  Record the plugin→core pairing in the commit message.

## Context

- `qwen38-27b/kernel/serve_qwen38.py` `DEFAULTS["vllm_tpu_version"]` pins
  `0.28.0`; `uv --exclude-newer` + env-dataset manifest date-gate cache
  reuse. PyPI shows a Sep-2026 `vllm-tpu` release (newer than 0.28.0).
- `patches/mtp-rollback-v0280.diff` is a port of `tpu-inference#3178`
  (GDN state-rollback for MTP speculative decoding). README says prefix
  caching is deliberately off for hybrid GDN on 0.28.0 with the fix landing
  upstream 2 days after release; `--no-async-scheduling` works around a
  `__delitem__` JSON-mode crash.

## Work items

1. Resolve latest `vllm-tpu` version + changelog: is `#3178` merged? Is the
   hybrid-GDN prefix-cache fix in? Is the `__delitem__`/async-scheduling
   bug fixed? (Web research; record URLs + versions in the commit message.)
2. Bump `vllm_tpu_version` in kernel `DEFAULTS`; update the
   `--exclude-newer` pin date and env-dataset manifest expectations.
3. If `#3178` is merged upstream: delete the patch path (`MTP_PATCH_B64`,
   `apply_mtp_patch()`, `patches/mtp-rollback-v0280.diff` handling) and
   keep MTP unconditionally on. If not merged: re-port the patch onto the
   new version (new `patches/mtp-rollback-vXXXX.diff`, re-embed via
   `qwen38-27b/tools/embed_patch.py`); keep fail-closed behavior
   (patch fails → `mtp_tokens=0`, never corrupt output).
4. Decide `--mtp` default (currently 3; PR #9 flipped to 0 citing a
   Hermes long-request `AttributeError` — do NOT flip without evidence on
   the new version; record the decision + rationale).
5. Decide prefix-caching flag (enable if upstream fixed it; else keep off
   with a versioned comment, not a 0.28.0 comment).
6. Decide `--trust-remote-code` (PR #1 injected it; add only if base
   `Qwen/Qwen3.8-27B` needs it on the new version).
7. Regenerate the Qwen notebook cell from the edited kernel.

## Non-goals

No weights-source changes (Stage B), no tunnel changes (Stage D), no doc
rewrites beyond versioned code comments (Stage F). Do not touch the Tauri
app. Do not add new speculative methods (PR #7 DFlash2 is rejected).

## Acceptance (semantic; live TPU deferred)

- [ ] `python3 -m py_compile launch.py qwen38-27b/kernel/serve_qwen38.py qwen38-27b/tools/embed_patch.py`
- [ ] Qwen notebook is valid JSON and its embedded kernel cell matches
      `serve_qwen38.py` byte-for-byte (modulo documented trailing newline).
- [ ] If patch kept: `embed_patch.py` roundtrip — decoded `MTP_PATCH_B64`
      equals the new `patches/*.diff`.
- [ ] No remaining `0.28.0` references in kernel/launcher (grep).
- [ ] Commit message records: new version, #3178 status, prefix-cache
      status, `__delitem__` status, mtp-default rationale, all with URLs.
- [ ] Deferred-live (list in commit): XLA cache-hit, MTP A/B exact-match,
      prefix-cache effect, JSON-mode+MTP crash test.
