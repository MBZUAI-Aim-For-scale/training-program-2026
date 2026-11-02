"""Check legacy configuration imports without loading weather dependencies."""

import subprocess
import sys
import unittest
from pathlib import Path


class ConfigImportTests(unittest.TestCase):
    def test_legacy_import_shares_canonical_module(self):
        project_root = Path(__file__).resolve().parents[1]
        script = """
import sys
import types

canonical = types.ModuleType('ai_weather.config')
canonical.PROJECT_ROOT = 'example-root'
sys.modules['ai_weather.config'] = canonical

import config
from config import PROJECT_ROOT

assert config is canonical
assert PROJECT_ROOT == 'example-root'
config.runtime_override = True
assert canonical.runtime_override is True
"""
        result = subprocess.run(
            [sys.executable, '-c', script],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
