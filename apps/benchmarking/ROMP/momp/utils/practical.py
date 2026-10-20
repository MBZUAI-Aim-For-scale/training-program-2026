import inspect
from importlib import resources
from pathlib import PurePosixPath, Path


#def set_dir(path_str):
def set_dir(path_str, work_dir=None):
    """
    Resolves a path string into an OS-agnostic Path or Traversable object.
    Checks local filesystem first, then falls back to package resources.
    """
    # 1. Convert to an OS-agnostic Path object (handles \ and / automatically)
    p = Path(path_str).expanduser()

    # 2. Priority 1: Check if the file exists locally (User Customization)
    #if p.exists():
    #    return p
        #return p.expanduser().resolve()

    # 3. Priority 2: Look inside the package resources
    # Use joinpath(*p.parts) to navigate the package structure correctly
    #package = "momp"
    #resource_target = resources.files(package).joinpath(*p.parts)

    if p.is_absolute():
        return p.resolve()

    elif not work_dir:
        #resource_target = Path(path_str).expanduser().resolve()
        resource_target = p.resolve()
    else:
        wd = Path(work_dir).expanduser().resolve()
        resource_target = (wd / p).resolve()
    
    # We return the 'Traversable' object. Both Path and Traversable have an .exists() method.
    return resource_target


def set_dir3(path):
    """
    Set absolute path for a resource inside the MOMP package
    Return a real filesystem Path for a resource inside the MOMP package.
    Accepts POSIX or Windows-style paths (e.g. 'data/input', 'data\\input').
    """
    package = "momp"
    base_dir = resources.files(package)

    # Normalize path into components (OS-independent)
    if isinstance(path, Path):
        parts = path.parts
    else:
        parts = PurePosixPath(str(path).replace("\\", "/")).parts

    for part in parts:
        target_dir = base_dir / part

    return resources.as_file(target_dir)


def set_dir2(folder):
    """
    original
    set absolute directory path for a specific folder in MOMP
    """
    package = "momp"
    base_dir = resources.files(package)
    target_dir = (base_dir / folder).resolve()

    return target_dir


# an safer version of set_dir, which can be called as p = set_dir("data/input")
# Keep temp dirs alive for program lifetime
_TEMP_DIRS = []

def set_dir_safe(path: str) -> Path:
    """
    Return a stable filesystem Path for a resource inside the MOMP package.
    Safe to assign: p = set_dir("data/input")
    Works on Windows, macOS, Linux, including zipped packages.
    """
    import tempfile
    import shutil
    import atexit

    package = "momp"

    # Normalize path into parts
    parts = PurePosixPath(path.replace("\\", "/")).parts

    resource = resources.files(package)
    for part in parts:
        resource = resource / part

    # If resource already exists on disk, return it directly
    try:
        real_path = Path(resource)
        if real_path.exists():
            return real_path
    except TypeError:
        pass  # Not a real filesystem path

    # Otherwise extract to a persistent temp directory
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
    """
    sig = inspect.signature(func)
    new_kwargs = dict(kwargs)

    for name, param in sig.parameters.items():
        if (
            param.kind is param.KEYWORD_ONLY
            and name in bound_args
            and name not in new_kwargs
        ):
            new_kwargs[name] = bound_args[name]

    return new_kwargs



