"""Select and check the scientific Python used by dashboard worker processes."""

import os
import sys
from pathlib import Path


def select_runtime_python(project_root):
    """Prefer the project environment without resolving its Python symlink."""
    override = os.environ.get("ROMP_PYTHON")
    local_python = Path(project_root) / ".venv" / "bin" / "python"
    candidate = Path(override).expanduser() if override else (
        local_python if local_python.is_file() else Path(sys.executable))
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise RuntimeError(f"Scientific Python is not executable: {candidate}. Check ROMP_PYTHON or the project .venv.")
    return str(candidate.absolute())


def check_runtime_python(python, env, cwd, spawn):
    """Probe binary dependencies in isolation using the same launcher as workers."""
    probe = (
        "import sys, numpy, pandas, xarray, matplotlib, scipy; "
        "from scipy import stats; "
        "stats.mannwhitneyu([0, 1], [1, 2]); "
        "print('Scientific Python:', sys.executable); "
        "print('SciPy:', scipy.__version__, scipy.__file__)"
    )
    output = []
    code, timed_out = spawn([python, "-c", probe], env, str(cwd), 30, output.append)
    if code != 0 or timed_out:
        detail = "Dependency check timed out." if timed_out else "\n".join(output)
        raise RuntimeError(
            f"Scientific Python dependency check failed for {python}. "
            "Start Streamlit with the project's .venv/bin/python, or set ROMP_PYTHON "
            "to a working environment. No benchmark was started.\n" + detail)
    return "\n".join(output)
