"""
momp/utils/practical.py
Path resolution, resource management, and argument restoration helpers.
"""

import inspect
import sys
import tempfile
import shutil
import atexit
from importlib import resources
from pathlib import PurePosixPath, Path

# Safe import for config and params
try:
    import config
except ImportError:
    try:
        from ai_weather import config
    except ImportError:
        _curr = Path(__file__).resolve()
        for _parent in _curr.parents:
            if (_parent / "ai_weather" / "config.py").exists() or (_parent / "config.py").exists():
                sys.path.insert(0, str(_parent / "ai_weather"))
                sys.path.insert(0, str(_parent))
                break
        import config

import momp.params as params


def set_dir(path_str, work_dir=None):
    """
    Resolves a path string into an OS-agnostic Path or URI string.
    Preserves GCS URIs (gs://) without path corruptions.
    """
    if not path_str:
        return ""

    s = str(path_str).strip()
    if config.is_uri(s):
        return f"gs://{config._strip(s)}"

    p = Path(s).expanduser()

    if p.is_absolute():
        return p.resolve()
    elif not work_dir:
        return p.resolve()
    else:
        wd = Path(work_dir).expanduser().resolve()
        return (wd / p).resolve()


def set_dir3(path):
    """
    Return a real filesystem Path for a resource inside the MOMP package.
    Accepts POSIX or Windows-style paths (e.g. 'data/input', 'data\\input').
    """
    package = "momp"
    base_dir = resources.files(package)

    if isinstance(path, Path):
        parts = path.parts
    else:
        parts = PurePosixPath(str(path).replace("\\", "/")).parts

    target_dir = base_dir
    for part in parts:
        target_dir = target_dir / part

    return resources.as_file(target_dir)


def set_dir2(folder):
    """
    Set absolute directory path for a specific folder inside the MOMP package.
    """
    package = "momp"
    base_dir = resources.files(package)
    target_dir = (base_dir / folder).resolve()
    return target_dir


_TEMP_DIRS = []


def set_dir_safe(path: str) -> Path:
    """
    Return a stable filesystem Path for a resource inside the MOMP package.
    Extracts zipped package resources into persistent temp directories if required.
    """
    package = "momp"

    parts = PurePosixPath(path.replace("\\", "/")).parts

    resource = resources.files(package)
    for part in parts:
        resource = resource / part

    try:
        real_path = Path(resource)
        if real_path.exists():
            return real_path
    except TypeError:
        pass

    temp_dir = tempfile.mkdtemp(prefix="momp_")
    _TEMP_DIRS.append(temp_dir)

    extracted_path = Path(temp_dir) / parts[-1]

    with resources.as_file(resource) as tmp:
        if tmp.is_dir():
            shutil.copytree(tmp, extracted_path)
        else:
            extracted_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp, extracted_path)

    return extracted_path


def restore_args(func, kwargs, bound_args):
    """
    Restore keyword-only parameters of `func` back into kwargs.
    Maps plural model lists (e.g., model_dir_list) to singular parameter names 
    and falls back to active attributes in momp.params when values are missing or None.
    """
    sig = inspect.signature(func)
    new_kwargs = dict(kwargs) if kwargs else {}

    # Extract dictionary view from bound_args or locals()
    if hasattr(bound_args, "arguments"):
        b_dict = bound_args.arguments
    elif isinstance(bound_args, dict):
        b_dict = bound_args
    else:
        b_dict = {}

    for name, param in sig.parameters.items():
        if name in ("kwargs", "self"):
            continue

        val = new_kwargs.get(name)
        if val is None:
            val = b_dict.get(name)

        # Map plural list parameters to singular names if missing or None
        if val is None:
            if name == "model_dir":
                dirs = new_kwargs.get("model_dir_list") or getattr(params, "model_dir_list", None)
                if dirs and len(dirs) > 0:
                    val = dirs[0] if isinstance(dirs, (list, tuple)) else dirs
            elif name == "file_pattern":
                pats = new_kwargs.get("file_pattern_list") or getattr(params, "file_pattern_list", None)
                if pats and len(pats) > 0:
                    val = pats[0] if isinstance(pats, (list, tuple)) else pats
            elif name == "model_var":
                vars_ = new_kwargs.get("model_var_list") or getattr(params, "model_var_list", None)
                if vars_ and len(vars_) > 0:
                    val = vars_[0] if isinstance(vars_, (list, tuple)) else vars_
            elif name in ("model", "model_name"):
                models = new_kwargs.get("model_list") or getattr(params, "model_list", None)
                if models and len(models) > 0:
                    val = models[0] if isinstance(models, (list, tuple)) else models

        # Fallback to momp.params
        if val is None and hasattr(params, name):
            val = getattr(params, name)

        # Fallback to function signature default if specified
        if val is None and param.default != inspect.Parameter.empty:
            val = param.default

        if val is not None:
            new_kwargs[name] = val

    return new_kwargs
