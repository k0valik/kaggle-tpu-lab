# Stage D — Tunnel + supply-chain hardening

## Goal

Close the real holes (unpinned `cloudflared`, secrets on ntfy/disk/logs)
and add an opt-in stable tunnel — without giving up the XLA cache or
bricking launches when GitHub is flaky.

## Context

Current holes: `cloudflared` fetched from `.../releases/latest/download/`
unpinned + unhashed in both kernels (Qwen in a background thread, failure
only logged); env-dataset copy is the only pin. `api_key` handling and
`ntfy` payload hygiene need review. References: PR #6 (qdubois: pinned
download-verify, `--host 127.0.0.1`, `redact()`, atomic `0600` state —
but wrongly deletes the XLA cache), PR #9 (KiVixx: named tunnel, secret
hygiene, foreground fetch), PR #5 (state `0600`).

## Work items

1. `cloudflared`: pinned version + sha256, download-to-temp + atomic
   install + pre-exec re-hash, foreground fetch with fail-fast message.
   Re-pin the digest at lift time (PR #6's `2026.8.3` digest is stale);
   keep a clear error (not silent dataset-binary fallback). Do NOT delete
   the env-dataset XLA cache path (PR #6's deletion rejected); add
   hash-check on the dataset copy if cheap.
2. vLLM `--host 127.0.0.1` in Qwen `server_args()` (and GLM if free);
   tunnel stays the public entry.
3. Secret hygiene: `redact()` on log/shell/pump paths; `publish(ready)`
   without `api_key` (endpoint/model only); launcher renders the key from
   local state; `ntfy` read cap (`1 MB`); `shutil.which("kaggle")`.
4. Atomic `0600` state/key helpers in `launch.py` (`write_state()`);
   `.gitignore` additions (`serve_config.json`, key files, `.venv/`,
   `work/`, bundle artifacts).
5. Opt-in stable tunnel: `cloudflare_hostname/token_secret/protocol` CFG
   + `--cloudflare-*` launcher flags, `TUNNEL_TOKEN` env (never argv),
   `watch_tunnel` with auto↔http2 backoff; random quick-tunnel stays the
   default. Stable API key from `KTL_API_KEY`-style secret as an option.
   Defer full HMAC `sign/verify` event protocol (needs versioning +
   migration story).
6. Regenerate affected notebook cells.

## PRs consulted (double-pass)

- [#6](https://github.com/k0valik/kaggle-tpu-lab/pull/6) (qdubois):
  Pinned-download + pre-exec re-hash logic, `--host 127.0.0.1`, `redact()`,
  atomic `0600` state, `.gitignore` entries. IGNORE env-dataset deletion,
  benchmark-deleting README. REVIEW (decide in build): full HMAC
  `sign/verify` protocol, `SESSION_DIR` isolation. Double-pass: re-check
  `sync_security.py`/`test_hardening.py` patterns for the Stage E harness.
  Double-pass V3 outcome: both CONFIRM-REJECT (narrow transport-only
  adversary, hard protocol break, single-tenant VM). Known-accepted:
  `benchmark`/`image-test` phases publish truncated model-output excerpts
  to the random-topic feed — HMAC wouldn't fix that (signed≠encrypted);
  topic entropy is the barrier; revisit only if excerpts carry sensitive
  content.
- [#9](https://github.com/k0valik/kaggle-tpu-lab/pull/9) (KiVixx):
  Named tunnel (`start_tunnel`/`watch_tunnel`), `TUNNEL_TOKEN` env,
  `KTL_API_KEY`-style secret, foreground fetch, `--use-xet`. IGNORE
  uncensored notebooks/hostnames, zh-TW runbook, queue monitor (macOS-only;
  at most re-check its abstract "never infer Running from Starting" rule).
- [#5](https://github.com/k0valik/kaggle-tpu-lab/pull/5) (Kitkitkittt):
  only the `download_checked` (resume+sha256+size) and `safe_extract`
  idioms, if they fit the pinned-download implementation.

## Non-goals

No cache deletion, no protocol breaks, no benchmark-deleting doc rewrites,
no Tauri changes.

## Acceptance (semantic; live TPU deferred)

- [ ] `py_compile`; notebooks valid JSON + in sync; no `latest/download`
      URLs remain in kernel code (grep); digests recorded with source +
      date in a comment.
- [ ] Unit-testable pieces covered locally: digest-verify helper (good
      vs tampered bytes), `redact()` over fixture log lines, atomic
      `0600` write under restrictive umask.
- [ ] Failure-mode reading: GitHub-flaky behavior documented (fail-hard
      vs fallback — record the chosen tradeoff explicitly).
- [ ] Deferred-live: tunnel comes up from pinned binary; named tunnel
      stable across relaunch; key never appears in ntfy/log captures.
