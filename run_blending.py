"""Run the configured onset blending pipeline with the project scientific Python."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "demos/demo4"
if (SOURCE_ROOT / "ai_weather/config.py").is_file():
    sys.path.insert(0, str(SOURCE_ROOT))

from ai_weather.runtime import check_runtime_python, select_runtime_python


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    in_process = "--in-process" in argv
    if in_process:
        argv.remove("--in-process")
    if not in_process:
        python = select_runtime_python(PROJECT_ROOT)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        if not any(flag in argv for flag in ("--help", "-h", "--dry_run")):
            check_runtime_python(python, env, PROJECT_ROOT)
        if Path(python).absolute() != Path(sys.executable).absolute():
            return subprocess.run(
                [python, "-u", str(Path(__file__).resolve()), "--in-process", *argv],
                cwd=Path.cwd(),
                env=env,
                check=False,
            ).returncode

    from apps.onset_blending.predict.run_operational_pipeline import (
        main as run_pipeline,
    )

    run_pipeline(argv)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] Onset blending failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
