"""Install the shared runtime plus the vendored ROMP package."""
from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup

momp_path = Path('apps/benchmarking/ROMP/momp')
if not momp_path.is_dir():
    raise FileNotFoundError("Could not find the 'momp' directory.")

# momp.app is a namespace package: find_packages alone silently omits its stages.
momp_subpackages = find_namespace_packages(where=str(momp_path))
runtime_packages = find_packages(include=['ai_weather', 'ai_weather.*', 'ui', 'ui.*'])
app_packages = find_namespace_packages(
    include=['apps', 'apps.*'],
    exclude=['apps.benchmarking.ROMP.momp', 'apps.benchmarking.ROMP.momp.*'],
)

setup(
    package_dir={'momp': str(momp_path), '': '.'},
    packages=runtime_packages + app_packages + ['momp'] + [f'momp.{p}' for p in momp_subpackages],
    py_modules=['config', 'run_momp', 'run_app', 'run_blending'],
)
