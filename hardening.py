"""Standard-library security helpers, also embedded in the standalone TPU script."""
import hashlib
import hmac
import json
import math
import os
import re
import tempfile
import time
import urllib.request
from pathlib import Path

# https://github.com/cloudflare/cloudflared/releases/tag/2026.8.3
# Update the version and digest together after checking the official release.
CLOUDFLARED_VERSION = "2026.8.3"
CLOUDFLARED_SHA256 = "f29324fe934d1e100617484c78deef803c4dc2cd351d645bbde42e96b4fccc5e"
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/download/"
    f"{CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
)


def download_cloudflared(destination):
    """Download to a private temporary file; install only after verification.

    Never trust an existing file or a dataset copy. Call in a private directory.
    A failed download or checksum raises, preventing tunnel startup.
    """
    destination = Path(destination)
    fd, tmp = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as out:
            with urllib.request.urlopen(CLOUDFLARED_URL, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    out.write(chunk)
            if not hmac.compare_digest(digest.hexdigest(), CLOUDFLARED_SHA256):
                raise RuntimeError("cloudflared SHA-256 mismatch; refusing to execute")
            out.flush()
            os.fsync(out.fileno())
        os.chmod(tmp, 0o700)
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def public_event(phase, extra):
    """Only allow structured progress metadata, never arbitrary logs or secrets."""
    if not isinstance(phase, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", phase):
        raise ValueError("Invalid progress phase")
    result = {"phase": phase}
    for field in ("elapsed_s", "startup_secs", "decode_tok_s", "max_model_len",
                  "max_num_seqs", "keepalive_min", "up_min", "served_min",
                  "secs", "entries", "rc"):
        value = extra.get(field)
        if type(value) in (int, float) and math.isfinite(value):
            result[field] = value
    if type(extra.get("covers_this_config")) is bool:
        result["covers_this_config"] = extra["covers_this_config"]
    endpoint = extra.get("endpoint")
    if isinstance(endpoint, str) and re.fullmatch(
            r"https://[a-z0-9-]+\.trycloudflare\.com/v1", endpoint):
        result["endpoint"] = endpoint
    # Known values only: arbitrary strings can contain credentials or prompt text.
    if extra.get("model") == "qwen3.8-27b":
        result["model"] = "qwen3.8-27b"
    if extra.get("step") in ("install", "server", "health-timeout", "tunnel"):
        result["step"] = extra["step"]
    return result


def sign_event(event, topic, event_key):
    if not event_key:
        raise ValueError("Missing progress signing key")
    payload = json.dumps({"topic": topic, "time": int(time.time()), "event": event},
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    signature = hmac.new(event_key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return json.dumps({"payload": payload, "signature": signature})


def verify_event(message, topic, event_key):
    """Authenticate progress before displaying it; legacy unsigned messages fail closed."""
    if not event_key or not isinstance(message, str) or len(message) > 16384:
        return None
    try:
        envelope = json.loads(message)
        payload, signature = envelope["payload"], envelope["signature"]
        if not isinstance(payload, str) or not isinstance(signature, str):
            return None
        expected = hmac.new(event_key.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return None
        data = json.loads(payload)
        stamp = data["time"]
        if type(stamp) is not int or not time.time() - 86400 <= stamp <= time.time() + 60:
            return None
        if data["topic"] != topic or not isinstance(data["event"], dict):
            return None
        event = data["event"]
        return stamp, public_event(event["phase"], event)
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError, RecursionError):
        return None


def write_private_state(path, state):
    """Atomically replace state with a mode-0600 file, without following symlinks."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".ktl-state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(state, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
