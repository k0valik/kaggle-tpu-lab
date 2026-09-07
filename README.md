# kaggle-tpu-lab — security hardening fork

Serve Qwen3.8-27B on a Kaggle TPU and use its OpenAI-compatible endpoint from your computer. Inference runs on Kaggle, not on your Mac.

This fork of [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) improves download verification and credential handling. It is not a security certification or an offline/private inference system. See [SECURITY.md](SECURITY.md) for the remaining trust boundaries.

## Changes from upstream

- Download `cloudflared` directly from Cloudflare's official **2026.8.3** release and verify a SHA-256 digest pinned in source before installation and again before execution. Never execute the dataset's binary or an existing `/tmp/cloudflared` file. Verification failure aborts startup.
- Send only allowlisted progress metadata to ntfy.sh. API keys, signing keys, log tails, free-form errors and model output are excluded. The public endpoint URL and timing information are still included.
- Authenticate progress with a separate, per-launch HMAC-SHA256 key. The launcher rejects unsigned, modified, wrong-session and expired messages. Signatures authenticate messages; they do not encrypt them.
- Display the inference key from local state, not from notifications. Replace local state atomically with an owner-only (`0600`) file. Redact the inference and signing keys from script-managed logs.
- Bind vLLM to `127.0.0.1`, with its API-key authentication enabled. Cloudflare Tunnel remains the public entry point.
- Compile TPU graphs in a fresh session directory. The author's environment dataset, executable and compiled cache are no longer attached or loaded. This costs startup time; upstream cached-start timings do not apply.

## Run from your terminal

Requires Python 3.9+ for the launcher, a Python version supported by the installed Kaggle CLI, and a Kaggle account with TPU access and available quota.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install kaggle
```

Configure the Kaggle CLI using your account's API credentials, then launch:

```bash
python launch.py serve --user YOUR_KAGGLE_USERNAME --text-only --keepalive-min 120
```

`--text-only` removes image inputs and their compilation cost. Omit it if you need vision. The launcher uploads a **private** script kernel with your generated inference key and progress-signing key embedded in its configuration. Keep that kernel private, including its source and any copies.

The local state file is `~/.kaggle-tpu-lab.json`. It contains secrets. The ready banner displays your endpoint and locally saved inference key. Configure your API client with these and model name `qwen3.8-27b`.

```bash
python launch.py status
python launch.py status -f
python launch.py stop
```

Ctrl-C detaches the launcher; the TPU keeps running. `stop` deletes the script kernel to terminate the session. Existing sessions launched with upstream's unsigned protocol must be stopped and relaunched to use this fork's progress verification.

`--fast-start` still skips initial graph compilation, but uncached request shapes compile on first use and can time out. This fork has not been benchmarked on a live TPU; allow for a cold start.

## Notebook

Upload `notebook/qwen38-tpu-serve.ipynb` to a **private** Kaggle notebook. Enable Internet and TPU VM v5e-8, and attach the weights dataset `rahim3/qwen3-8-27b-bf16`. Do not attach the environment dataset.

The notebook includes the same hardened standalone script. With its default empty `ntfy_topic`, it sends no ntfy notifications. It saves its generated inference key to an owner-only file under the session's temporary directory and prints that file's location, not the key. Retrieve it privately using a terminal in the same Kaggle session. The CLI workflow above is easier because it already has the key locally. Never publish a notebook containing credentials or credential outputs.

## Development and verification

The security helpers are maintained in `hardening.py` and embedded in the standalone kernel so Kaggle needs only one script. After editing them or the kernel:

```bash
python tools/sync_security.py
python -m unittest discover -s tests -v
python -m py_compile hardening.py launch.py kernel/serve_qwen38.py
```

Tests cover credential exclusion, signed progress, tampering, malformed messages, failed downloads, state-file permissions, private kernel configuration, loopback binding, and notebook synchronization. They mock Kaggle and inference; they do not prove TPU runtime compatibility.

To update Cloudflare's binary pin, obtain both version and digest from the [official release](https://github.com/cloudflare/cloudflared/releases), update `hardening.py`, regenerate the embedded copies, and verify the real downloaded bytes. Do not substitute a floating `latest` URL or a checksum from the dataset.

The upstream `build-env` command remains available for maintainers to export newly compiled graphs and metadata, but serving in this fork does not import those caches and the bundle no longer includes a tunnel executable.

## Credits and license

Original serving implementation and TPU patch: [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab). Inference stack: vLLM and tpu-inference. Model: Qwen. Hardware: Kaggle.

Repository code remains MIT licensed; see [LICENSE](LICENSE). Model weights retain their upstream license. Upstream's performance figures describe its configuration and have not been revalidated for this fork.
