# Security model

This fork reduces specific credential-disclosure and executable-supply risks. It does not make Kaggle-hosted inference private, end-to-end encrypted, independently audited, or safe against compromised providers or dependencies.

## Fixed boundaries

The inference API key is generated on the client, saved in a mode-0600 local state file and included in the private Kaggle kernel configuration. It is never intentionally sent to ntfy.sh. The launcher displays the local copy. Notebook-only runs save their generated key in a private temporary file instead of printing it into notebook output.

A separate random progress key signs each event using HMAC-SHA256. The signed payload includes the launch-specific topic and timestamp. The receiver verifies the signature before displaying progress, ignores unsigned legacy messages and rejects messages older than 24 hours or more than 60 seconds in the future. Same-session messages can still be replayed within that window; signatures do not guarantee freshness or availability. Events cannot cause the launcher to execute commands or make authenticated requests to a supplied endpoint.

An allowlist excludes arbitrary strings, error/log payloads, keys, prompts and model responses from notifications. The public Cloudflare endpoint, known model name, event phase and numeric progress information remain visible to ntfy and anyone who knows the topic. Notifications are signed, not encrypted. ntfy can drop, delay or replay progress. If it fails, consult your private Kaggle session.

The tunnel executable comes from the pinned official Cloudflare GitHub release over HTTPS. Its SHA-256 must match the digest committed in `hardening.py`. Downloads use a private temporary file and are installed atomically after verification. A fresh private session directory prevents accidental reuse of an old `/tmp/cloudflared`; the digest is checked again before execution. SHA-256 pinning trusts the release and digest chosen during review; it is not an independent publisher signature or a guarantee of vulnerability-free code.

The author's compiled XLA cache and environment dataset are not loaded, even if attached to a notebook. Graphs compile in a fresh private session directory. This avoids trusting third-party compiled artifacts and avoids extracting their archives.

## Remaining trust and exposure

- **Kaggle** runs the model and can access the private kernel configuration, memory, prompts and files. Uploaded source contains both generated keys; anyone granted access to that source can recover them. Local mode-0600 permissions do not protect against your own account, administrators or malware.
- **Cloudflare** terminates the public HTTPS connection. Requests pass through its tunnel to Kaggle. This is not end-to-end encryption between your computer and the model process.
- **Public API**: the endpoint is Internet reachable and protected by a bearer key. There is no additional identity gateway, per-client quota, request-rate policy or comprehensive endpoint authorization audit. Treat keys as sensitive and stop/relaunch to rotate them. Unauthenticated denial-of-service remains possible.
- **Dependencies**: the pinned `vllm-tpu` version and bundled patch remain, but `uv` and all transitive packages are not locked with verified hashes. Installing a package executes trusted third-party code. Python package indexes, publishers and Kaggle's base image remain in the trust chain.
- **Weights**: the original author's Kaggle mirror is still the default weight/tokenizer/config source; it has not been compared file-by-file with a pinned official Qwen revision. The fallback downloads from Qwen on Hugging Face without pinning a model revision. No `trust_remote_code` flag is added, but model assets and their parsers still require trust.
- **Logs**: this fork redacts literal inference and progress keys in its managed output and excludes logs from ntfy. This is not a general PII scrubber and does not guarantee that every dependency avoids logging prompts or encoded secrets. vLLM's command arguments also contain its API key inside the Kaggle environment.
- **Coding agents**: an inference endpoint can return tool calls. Keep the consuming agent's filesystem, network and command permissions constrained to the intended project.

## Validation and deployment status

Local tests exercise security behavior without installing the inference stack or starting a Kaggle job. The pinned official Linux binary was downloaded and its digest verified; it was not executed on the ARM Mac. A private live Kaggle smoke test is still required to validate TPU startup and a request through the tunnel. No claim is made that all dependencies, binary contents or model files have been audited.
