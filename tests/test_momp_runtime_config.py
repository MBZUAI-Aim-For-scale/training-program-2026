"""Exercise runner overrides without importing weather-data dependencies."""
import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class RuntimeConfigTests(unittest.TestCase):
    def test_onset_overrides_and_explicit_no_mask(self):
        tree = ast.parse(Path('run_momp.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'apply_env_vars_to_cfg')
        namespace = {'os': os, 'ast': ast, 'resolve_path': lambda p: '/resolved/' + p}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'run_momp.py', 'exec'), namespace)
        cfg = SimpleNamespace(wet_threshold=20, nc_mask='old.nc', thresh_file=None)
        with patch.dict(os.environ, {'ROMP_WET_THRESHOLD': '35',
                                     'ROMP_DRY_EXTENT': '0',
                                     'ROMP_AREA_MASK_FILE': '',
                                     'ROMP_ONSET_THRESHOLD_FILE': 'thresholds.csv',
                                     'ROMP_BASELINE_FORECAST': 'climatology'}, clear=True):
            namespace['apply_env_vars_to_cfg'](cfg)
        self.assertEqual(cfg.wet_threshold, 35)
        self.assertEqual(cfg.dry_extent, 0)
        self.assertIsNone(cfg.nc_mask)
        self.assertEqual(cfg.thresh_file, '/resolved/thresholds.csv')


if __name__ == '__main__':
    unittest.main()
