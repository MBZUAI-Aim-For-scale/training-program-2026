"""Regression cases for ROMP utility paths missed by the main workflow."""

import ast
from pathlib import Path
import tempfile
import unittest

import pandas as pd


ROOT = Path('apps/benchmarking/ROMP/momp')


def load_function(relative, name, **namespace):
    tree = ast.parse((ROOT / relative).read_text())
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    return namespace[name]


class UtilityAuditTests(unittest.TestCase):
    def test_day_of_year_labels_follow_the_requested_calendar(self):
        convert = load_function('graphics/func_map.py', 'doy_to_date_string', pd=pd)
        self.assertEqual(convert(60, 2024), '29/02')
        self.assertEqual(convert(60, 2023), '01/03')

    def test_missing_netcdf_error_identifies_the_searched_directory(self):
        inspect = load_function('utils/printing.py', 'print_data_info')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, directory):
                inspect(Path(directory))

    def test_nested_dictionary_reports_all_keys_at_each_depth(self):
        analyze = load_function('io/output.py', 'analyze_nested_dict')
        data = {'a': {'x': {'score': 1}}, 'b': {'y': {'score': 2, 'other': 3}}}
        self.assertEqual(analyze(data), (3, [['a', 'b'], ['x', 'y'], ['score', 'other']]))
        self.assertEqual(analyze({}), (0, []))
        with self.assertRaises(TypeError):
            analyze([])
