#!/usr/bin/env python3
"""Convert '# %%' percent-format .py files to .ipynb (stdlib only).

Usage: py2nb.py lab01.py [lab02.py ...]
A cell starts at '# %%' (code) or '# %% [markdown]' (markdown; body lines are '# ' prefixed).
"""
import json
import sys
from pathlib import Path


def convert(path: Path) -> Path:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        src = [l + "\n" for l in lines]
        if src:
            src[-1] = src[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in path.read_text().splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        elif kind is not None:
            buf.append(line)
    flush()

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = path.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1) + "\n")
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("wrote", convert(Path(p)))
