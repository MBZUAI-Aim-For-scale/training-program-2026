"""Installation must include the actual runner, GUI and scientific modules."""
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import unittest
from unittest.mock import patch


class PackagingAuditTests(unittest.TestCase):
    def test_setup_contains_runtime_packages_and_actual_root_modules(self):
        # Exercise setup metadata without requiring the optional build backend.
        def discover(where='.', include=None, **kwargs):
            roots = [Path(where, 'ai_weather'), Path(where, 'ui')] if include else [Path(where)]
            paths = [path for root in roots for path in [root, *root.rglob('*')]]
            return [str(path.relative_to(where)).replace('/', '.') for path in paths
                    if path.is_dir() and (path / '__init__.py').is_file()
                    and path != Path(where)]
        def namespaces(where='.', include=None, exclude=(), **kwargs):
            import fnmatch
            root = Path(where, 'apps') if include else Path(where)
            paths = [root, *root.rglob('*')]
            names = [str(path.relative_to(where)).replace('/', '.') for path in paths
                     if path.is_dir() and path != Path(where)]
            return [name for name in names if (not include or any(fnmatch.fnmatchcase(name, pattern)
                                                                for pattern in include))
                    and not any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude)]
        setup = MagicMock()
        backend = SimpleNamespace(setup=setup, find_packages=discover,
                                  find_namespace_packages=namespaces)
        with patch.dict(sys.modules, {'setuptools': backend}):
            runpy.run_path('setup.py', run_name='audit_setup')
        options = setup.call_args.kwargs
        for package in ('ai_weather', 'ui', 'apps.forecasting', 'momp', 'momp.app'):
            self.assertIn(package, options['packages'])
        self.assertEqual(options['package_dir'].get('', '.'), '.')
        self.assertIn('run_momp', options['py_modules'])
        self.assertNotIn('dataset', options['py_modules'])

    def test_installed_momp_package_is_found_without_vendored_apps_tree(self):
        import ast
        import tempfile
        tree = ast.parse(Path('ai_weather/config.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_find_momp_package_dir')
        namespace = {'Path': Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'config', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installed = root / 'site-packages/momp'
            installed.mkdir(parents=True)
            (installed / 'driver.py').touch()
            spec = SimpleNamespace(origin=str(installed / '__init__.py'))
            with patch('importlib.util.find_spec', return_value=spec):
                self.assertEqual(namespace['_find_momp_package_dir'](root / 'apps'), installed)
                vendored = root / 'apps/benchmarking/ROMP/momp'
                vendored.mkdir(parents=True)
                (vendored / 'driver.py').touch()
                self.assertEqual(namespace['_find_momp_package_dir'](root / 'apps'), vendored)
