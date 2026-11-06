"""Select and check the scientific Python used by benchmark processes."""

import os
from pathlib import Path
import subprocess
import sys


def select_runtime_python(project_root):
    """Prefer a project environment, including above a relocated demo directory.

    Stop at the nearest repository/package boundary so a flattened Docker app
    retains its own environment. Never resolve the Python executable symlink:
    doing so bypasses the virtual environment's site-packages.
    """
    override = os.environ.get("ROMP_PYTHON")
    if override:
        candidate = Path(override).expanduser()
    else:
        root = Path(project_root).expanduser().absolute()
        search_roots = []
        for parent in (root, *root.parents):
            search_roots.append(parent)
            if (parent / ".git").exists() or (parent / "pyproject.toml").is_file():
                break
        else:
            search_roots = [root]
        local_python = next((parent / ".venv/bin/python" for parent in search_roots
                             if (parent / ".venv/bin/python").is_file()), None)
        candidate = local_python or Path(sys.executable)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise RuntimeError(f"Scientific Python is not executable: {candidate}. Check ROMP_PYTHON or the project .venv.")
    return str(candidate.absolute())


def _spawn_dependency_probe(argv, env, cwd, timeout, on_line):
    try:
        result = subprocess.run(argv, env=env, cwd=cwd, timeout=timeout,
                                capture_output=True, text=True, check=False)
    except subprocess.TimeoutExpired:
        return 124, True
    for line in (result.stdout + result.stderr).splitlines():
        on_line(line)
    return result.returncode, False


def check_runtime_python(python, env, cwd, spawn=None):
    """Probe binary dependencies in isolation using the same launcher as workers."""
    probe = (
        "import sys, numpy, pandas, xarray, matplotlib, scipy; "
        "from scipy import stats; "
        "stats.mannwhitneyu([0, 1], [1, 2]); "
        "print('Scientific Python:', sys.executable); "
        "print('SciPy:', scipy.__version__, scipy.__file__)"
    )
    output = []
    spawn = spawn or _spawn_dependency_probe
    code, timed_out = spawn([python, "-c", probe], env, str(cwd), 30, output.append)
    if code != 0 or timed_out:
        detail = "Dependency check timed out." if timed_out else "\n".join(output)
        raise RuntimeError(
            f"Scientific Python dependency check failed for {python}. "
            "Use the project's .venv/bin/python, or set ROMP_PYTHON "
            "to a working environment. No benchmark was started.\n" + detail)
    return "\n".join(output)
