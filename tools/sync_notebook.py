#!/usr/bin/env python3
"""Single-source sync: kernel script(s) -> notebook ``%%writefile`` cells.

The kernel ``serve_*.py`` files are the source of truth (see AGENTS.md);
each recipe's notebook is a generated copy. This tool refreshes the
generated cells in place so hand-authored cells (preflight, config,
launch notes) are never touched:

    tools/sync_notebook.py qwen38-27b          # refresh Qwen notebook cells
    tools/sync_notebook.py glm53-flash         # refresh GLM notebook cells
    tools/sync_notebook.py qwen38-27b --check  # CI: exit 0 in sync, 1 stale

Managed cells per recipe (located BY CONTENT, never by index — a hardcoded
cell index once overwrote the wrong cell, so every managed cell is found by
matching its first line / marker text):

- one ``%%writefile <kernel-name>`` code cell per kernel script, found by
  matching the filename in the cell's first line; its body is replaced with
  the current file content;
- one engine cell per ``<recipe>/engine/*/__init__.py`` package (GLM only;
  Qwen has no engine dir), found by its ``os.makedirs("<pkg>", ...)``
  marker and regenerated with ``pack_notebook.engine_cell`` (reused, not
  reinvented — GLM output stays byte-identical to ``pack_notebook.py``).

Every code cell is also normalized to ship clean (``outputs == []``,
``execution_count is None``); anything else counts as stale. Notebooks are
written back canonically with
``json.dumps(nb, indent=1, ensure_ascii=False) + "\\n"`` — any other dump
settings churn the diff.

Exit codes: 0 = synced / in sync; 1 = stale cells (``--check`` only);
2 = usage or sync error (missing notebook/kernel/cell).
"""
import argparse
import json
import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
from pack_notebook import engine_cell  # noqa: E402  (reuse GLM logic verbatim)

RECIPES = ("qwen38-27b", "glm53-flash")

WRITEFILE_RE = re.compile(r"^%%writefile\s+(\S+)\s*$")


def first_line(cell):
    src = "".join(cell.get("source", []))
    lines = src.splitlines()
    return lines[0] if lines else ""


def find_writefile_cell(cells, filename):
    """Index of the code cell whose first line is ``%%writefile <filename>``."""
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        m = WRITEFILE_RE.match(first_line(c))
        if m and m.group(1) == filename:
            return i
    return None


def find_engine_cell(cells, pkg):
    """Index of the code cell generated for engine package ``pkg``."""
    marker = f'os.makedirs("{pkg}", exist_ok=True)'
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        if marker in "".join(c.get("source", [])):
            return i
    return None


def to_lines(text):
    return text.splitlines(keepends=True)


def sync_recipe(recipe, check=False):
    folder = ROOT / recipe
    nbs = sorted((folder / "notebook").glob("*.ipynb")) if (folder / "notebook").is_dir() else []
    if len(nbs) != 1:
        return [f"error: expected exactly one notebook in {folder / 'notebook'}, found {len(nbs)}"], 2
    kernels = sorted((folder / "kernel").glob("serve_*.py")) if (folder / "kernel").is_dir() else []
    if len(kernels) != 1:
        return [f"error: expected exactly one kernel in {folder / 'kernel'}, found {len(kernels)}"], 2
    nb_path, kernel = nbs[0], kernels[0]
    try:
        nb = json.loads(nb_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"error: cannot read {nb_path}: {e}"], 2
    cells = nb.get("cells", [])

    engine_inits = sorted((folder / "engine").glob("*/__init__.py")) if (folder / "engine").is_dir() else []

    stale, errors = [], []

    # 1. Kernel -> %%writefile cell (located by filename in first line).
    try:
        kernel_text = kernel.read_text()
    except OSError as e:
        return [f"error: cannot read {kernel}: {e}"], 2
    idx = find_writefile_cell(cells, kernel.name)
    if idx is None:
        errors.append(f"error: {nb_path.name} has no %writefile cell for {kernel.name}")
    else:
        want = to_lines(f"%%writefile {kernel.name}\n" + kernel_text)
        if list(cells[idx].get("source", [])) != want:
            stale.append(f"{nb_path.name}:cell[{idx}] %%writefile {kernel.name}")
            if not check:
                cells[idx]["source"] = want

    # 2. Engine package(s) -> engine cell(s) (GLM; Qwen has no engine dir).
    for init in engine_inits:
        pkg = init.parent.name
        try:
            want_text = engine_cell(init.parent, pkg)
        except AssertionError as e:
            errors.append(f"error: engine cell for {pkg}: {e}")
            continue
        idx = find_engine_cell(cells, pkg)
        if idx is None:
            errors.append(f"error: {nb_path.name} has no engine cell for package {pkg!r}")
            continue
        want = to_lines(want_text)
        if list(cells[idx].get("source", [])) != want:
            stale.append(f"{nb_path.name}:cell[{idx}] engine package {pkg}")
            if not check:
                cells[idx]["source"] = want

    # 3. All code cells ship clean: no outputs, null execution_count.
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        if c.get("outputs", []) != [] or c.get("execution_count") is not None:
            stale.append(f"{nb_path.name}:cell[{i}] outputs/execution_count not clean")
            if not check:
                c["outputs"] = []
                c["execution_count"] = None

    if errors:
        return errors, 2
    if check:
        if stale:
            return [f"stale: {s}" for s in stale], 1
        return [f"{recipe}: {nb_path.name} in sync"], 0
    if stale:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        return [f"updated: {s}" for s in stale], 0
    return [f"{recipe}: {nb_path.name} already in sync"], 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Sync kernel script(s) into notebook %%writefile cells "
                    "(kernel files are the source of truth; --check for CI).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe", choices=RECIPES, help="which recipe folder to sync")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 0 if in sync, 1 listing stale cells")
    args = ap.parse_args(argv)
    messages, code = sync_recipe(args.recipe, check=args.check)
    for m in messages:
        print(m, file=sys.stderr if code == 2 else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
