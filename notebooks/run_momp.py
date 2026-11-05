#!/usr/bin/env python3
"""Notebook-compatible entry point for the canonical project ROMP runner.

Keep configuration merging, validation, and scientific execution in run_momp.py
at the project root so notebook, dashboard, and CLI results use the same code.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("AI_WEATHER_REPO") or Path(__file__).resolve().parents[1])
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_weather.runtime import check_runtime_python, select_runtime_python


def _canonical_runner():
    name = "_ai_weather_canonical_run_momp"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "run_momp.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def execute_momp_benchmark(mode="det", model_key=None, in_process=False):
    python = select_runtime_python(PROJECT_ROOT)
    if python != os.path.abspath(sys.executable):
        cmd = [python, str(Path(__file__).resolve()), "--mode", mode]
        if model_key:
            cmd.extend(["--model", model_key])
        if in_process:
            cmd.append("--in-process")
        print(f"Using scientific Python: {python}", flush=True)
        return subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=os.environ.copy(), check=False).returncode
    # Preserve the notebook API's dependency error before importing its pipeline.
    print(check_runtime_python(python, os.environ.copy(), PROJECT_ROOT), flush=True)
    return _canonical_runner().execute_momp_benchmark(mode, model_key, in_process)


def __getattr__(name):
    # Older notebooks import helper functions/constants from this module.
    return getattr(_canonical_runner(), name)


def main():
    return _canonical_runner().main()


if __name__ == "__main__":
    main()
