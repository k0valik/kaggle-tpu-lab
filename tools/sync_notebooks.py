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


if __name__ == "__main__":
    main()
