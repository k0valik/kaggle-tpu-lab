#!/usr/bin/env python3
"""Embed shared helpers and synchronize the notebook's standalone script.

Run after changing hardening.py or kernel/serve_qwen38.py.
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
script = root / "kernel/serve_qwen38.py"
source = script.read_text()
start = source.index("# BEGIN EMBEDDED HARDENING")
end = source.index("# END EMBEDDED HARDENING", start)
source = (source[:start] + "# BEGIN EMBEDDED HARDENING\n"
          + (root / "hardening.py").read_text() + "\n" + source[end:])
script.write_text(source)
path = root / "notebook/qwen38-tpu-serve.ipynb"
notebook = json.loads(path.read_text())
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        if "".join(cell["source"]).startswith("%%writefile serve_qwen38.py"):
            cell["source"] = ("%%writefile serve_qwen38.py\n" + source).splitlines(True)
        cell["outputs"] = []
        cell["execution_count"] = None
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
