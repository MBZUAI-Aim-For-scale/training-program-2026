"""Repository and flattened Docker layouts expose the same runtime entrypoints."""

import ast
from contextlib import chdir
import fnmatch
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]


def discover(where=".", include=("*",), exclude=(), namespaces=False):
    root = Path(where)
    paths = [root, *root.rglob("*")]
    names = [str(path.relative_to(root)).replace("/", ".") for path in paths
             if path != root and path.is_dir() and "__pycache__" not in path.parts
             and (namespaces or (path / "__init__.py").is_file())]
    return [name for name in names if any(fnmatch.fnmatchcase(name, pattern) for pattern in include)
            and not any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude)]


class Demo4RelocationPackagingTests(unittest.TestCase):
    def metadata(self, root):
        backend = ModuleType("setuptools")
        backend.setup = MagicMock()
        backend.find_packages = discover
        backend.find_namespace_packages = lambda **kwargs: discover(namespaces=True, **kwargs)
        with chdir(root), patch.dict(sys.modules, {"setuptools": backend}):
            runpy.run_path(str(root / "setup.py"), run_name="relocation_setup_test")
        return backend.setup.call_args.kwargs

    def assert_runtime_paths(self, root, options):
        for package, filename in (("ai_weather", "config.py"), ("ui", "app.py"),
                                  ("apps.forecasting", "runs.py"), ("momp", "driver.py")):
            self.assertIn(package, options["packages"])
            self.assertTrue((root / options["package_dir"][package] / filename).is_file(), package)
        self.assertIn("momp.app", options["packages"])
        for module in ("config", "run_momp", "run_app", "run_blending"):
            self.assertIn(module, options["py_modules"])
            self.assertTrue((root / options["package_dir"][""] / f"{module}.py").is_file(), module)

    def test_repo_metadata_finds_moved_runtime_and_root_blending(self):
        options = self.metadata(ROOT)
        self.assert_runtime_paths(ROOT, options)
        self.assertIn("apps.onset_blending", options["packages"])
        self.assertEqual(options["package_dir"]["apps.onset_blending"], "apps/onset_blending")
        self.assertEqual(options["package_dir"]["momp"], "demos/demo4/apps/benchmarking/ROMP/momp")

    def test_flattened_metadata_keeps_docker_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "setup.py", root / "setup.py")
            for relative in ("ai_weather/__init__.py", "ai_weather/config.py", "ui/__init__.py",
                             "ui/app.py", "apps/forecasting/runs.py", "apps/forecasting/__init__.py",
                             "apps/benchmarking/ROMP/momp/__init__.py",
                             "apps/benchmarking/ROMP/momp/driver.py",
                             "apps/benchmarking/ROMP/momp/app/stage.py"):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            for module in ("config", "run_momp", "run_app", "run_blending"):
                (root / f"{module}.py").touch()
            options = self.metadata(root)
            self.assert_runtime_paths(root, options)
            self.assertEqual(options["package_dir"]["momp"], "apps/benchmarking/ROMP/momp")

    def test_launcher_finds_moved_and_flattened_app(self):
        namespace = runpy.run_path(str(ROOT / "run_app.py"), run_name="relocation_launcher_test")
        main = namespace["main"]
        for relative in ("demos/demo4/ui/app.py", "ui/app.py"):
            with self.subTest(layout=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                app = root / relative
                app.parent.mkdir(parents=True)
                app.touch()
                launch = MagicMock()
                with patch.dict(main.__globals__, {"__file__": str(root / "run_app.py"),
                                                   "run_streamlit": launch}), \
                     patch.object(sys, "argv", ["run_app.py"]), \
                     patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
                    self.assertEqual(main(), 0)
                self.assertEqual(launch.call_args.args[0], app.resolve())

    def test_config_shim_aliases_canonical_moved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "config.py", root / "config.py")
            package = root / "demos/demo4/ai_weather"
            package.mkdir(parents=True)
            (package / "__init__.py").touch()
            (package / "config.py").write_text("DATA_DIR = 'relocated-test-data'\n")
            result = subprocess.run([sys.executable, "-c",
                                     "import config; from ai_weather import config as canonical; "
                                     "assert config is canonical; assert config.DATA_DIR == 'relocated-test-data'"],
                                    cwd=root, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_root_runners_bootstrap_moved_runtime_before_imports(self):
        for filename, stop_name in (("run_momp.py", "LOCAL_ROMP_ROOT"), ("run_blending.py", "main")):
            with self.subTest(runner=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                package = root / "demos/demo4/ai_weather"
                package.mkdir(parents=True)
                (package / "__init__.py").touch()
                (package / "config.py").write_text("DATA_DIR = 'relocated-test-data'\n")
                (package / "runtime.py").write_text(
                    "def check_runtime_python(*args): pass\ndef select_runtime_python(*args): pass\n")
                nodes = []
                for node in ast.parse((ROOT / filename).read_text()).body:
                    if isinstance(node, ast.FunctionDef) and node.name == stop_name:
                        break
                    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                            and target.id == stop_name for target in node.targets):
                        break
                    nodes.append(node)
                program = root / filename
                program.write_text(ast.unparse(ast.Module(body=nodes, type_ignores=[])))
                result = subprocess.run([sys.executable, str(program)], cwd=root,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
