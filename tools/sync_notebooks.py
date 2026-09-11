#!/usr/bin/env python3
"""Refresh the embedded serving script and clear outputs in repository notebooks."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernel" / "serve_qwen38.py"
NOTEBOOKS = sorted((ROOT / "notebook").glob("*.ipynb"))


def source_lines(text):
    return text.splitlines(keepends=True)


def sync(path, kernel_source):
    notebook = json.loads(path.read_text())
    embedded = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = []
        source = "".join(cell.get("source", []))
        if source.startswith("%%writefile serve_qwen38.py"):
            cell["source"] = source_lines("%%writefile serve_qwen38.py\n" + kernel_source)
            embedded += 1
    if embedded != 1:
        raise RuntimeError(f"{path}: expected one embedded kernel cell, found {embedded}")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")


def main():
    kernel_source = KERNEL.read_text()
    for path in NOTEBOOKS:
        sync(path, kernel_source)
        print(f"synced {path.relative_to(ROOT)}")
    # A ready-to-import personal Draft, with no credentials or execution outputs.
    notebook = json.loads((ROOT / "notebook/qwen38-tpu-serve-uncensored.ipynb").read_text())
    config_cell = next(c for c in notebook["cells"]
                       if "".join(c.get("source", [])).startswith("%%writefile serve_config.json"))
    config = json.loads("".join(config_cell["source"]).split("\n", 1)[1])
    config.update(weights_dataset="king88888888/qwen38-27b-uncensored-bf16",
                  cloudflare_hostname="qwen.aceinifnity.com", cloudflare_protocol="auto")
    config_cell["source"] = source_lines("%%writefile serve_config.json\n" + json.dumps(config, indent=2) + "\n")
    notebook["cells"][0]["source"] = source_lines(
        "# Qwen3.8 27B — Interactive Draft\n\n"
        "Use Chrome and Run All in the Draft editor. Accelerator: TPU VM v5e-8; Internet: ON.\n\n"
        "Attach `king88888888/qwen38-27b-uncensored-bf16` (model weights) and "
        "`rahim3/qwen38-tpu-env-v5e8` (runtime/cache) before starting. "
        "Enable Kaggle Secrets `KTL_API_KEY` and `CF_TUNNEL_TOKEN`.\n\n"
        "The serving script is embedded below; no extra patch cells are needed. "
        "Every launch downloads and validates a fresh cloudflared. "
        "Wait for warm-up and benchmark before testing Hermes.\n\n"
        "Stopping the Python cell or closing Chrome does not prove the TPU session stopped. "
        "Use Active Events → Stop Session and confirm zero active events.\n")
    target = ROOT / "notebook/qwen38-tpu-draft.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"generated {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
