"""Install the shared runtime plus the vendored ROMP package."""
from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup

source_root = Path('demos/demo4')
if not (source_root / 'apps/benchmarking/ROMP/momp/driver.py').is_file():
    source_root = Path('.')
momp_path = source_root / 'apps/benchmarking/ROMP/momp'
if not (momp_path / 'driver.py').is_file():
    raise FileNotFoundError("Could not find the 'momp' directory.")

# momp.app is a namespace package: find_packages alone silently omits its stages.
momp_subpackages = find_namespace_packages(where=str(momp_path))
runtime_packages = find_packages(where=str(source_root), include=['ai_weather', 'ai_weather.*', 'ui', 'ui.*'])
root_app_packages = find_namespace_packages(
    include=['apps', 'apps.*'],
    exclude=['apps.benchmarking.ROMP.momp', 'apps.benchmarking.ROMP.momp.*'],
)
source_app_packages = find_namespace_packages(
    where=str(source_root),
    include=['apps', 'apps.*'],
    exclude=['apps.benchmarking.ROMP.momp', 'apps.benchmarking.ROMP.momp.*'],
)
app_packages = list(dict.fromkeys(root_app_packages + source_app_packages))
package_dir = {'': '.', 'momp': str(momp_path)}
package_dir.update({name: name.replace('.', '/') for name in root_app_packages})
package_dir.update({name: str(source_root / name.replace('.', '/'))
                    for name in runtime_packages + source_app_packages})

setup(
    package_dir=package_dir,
    packages=runtime_packages + app_packages + ['momp'] + [f'momp.{p}' for p in momp_subpackages],
    py_modules=['config', 'run_momp', 'run_app', 'run_blending'],
)
