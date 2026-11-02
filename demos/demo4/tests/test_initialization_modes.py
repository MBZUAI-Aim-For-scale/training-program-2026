"""Initialization controls select real dates without invalid calendar operations."""

import ast
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


class InitializationModeTests(unittest.TestCase):
    def setUp(self):
        path = Path("apps/benchmarking/ROMP/momp/io/input.py")
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"get_initialization_dates", "get_initialization_dates_monthly"}]
        ns = {"np": np, "pd": pd, "datetime": datetime,
              "params": SimpleNamespace(date_filter_year=2024, init_type="weekly", init_days=(0, 3),
                                         start_date=(2003, 5, 1), end_date=(2024, 7, 31))}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        self.select = ns["get_initialization_dates"]

    def test_daily_uses_every_target_year_date_without_weekday_filter(self):
        actual = self.select(2023, init_type="daily", init_days=(),
                             start_date=(2023, 5, 1), end_date=(2023, 5, 7))
        pd.testing.assert_index_equal(actual, pd.date_range("2023-05-01", "2023-05-07"))

    def test_weekly_preserves_reference_year_weekday_anchoring(self):
        actual = self.select(2003, init_type="weekly", date_filter_year=2024, init_days=(0, 3),
                             start_date=(2003, 5, 1), end_date=(2023, 5, 10))
        # May 2, 6 and 9 are Thu/Mon/Thu in the reference year, not in 2003.
        pd.testing.assert_index_equal(actual, pd.DatetimeIndex(["2003-05-02", "2003-05-06", "2003-05-09"]))

    def test_invalid_mode_empty_or_invalid_days_are_rejected(self):
        for mode, days in (("weekly", ()), ("weekly", (7,)), ("weekly", (True,)),
                           ("monthly", (0,)), ("monthly", (32,)), ("monthly", ())):
            with self.subTest(mode=mode, days=days), self.assertRaises(ValueError):
                self.select(2024, init_type=mode, init_days=days)
        with self.assertRaises(ValueError):
            self.select(2024, init_type="unexpected")

    def test_monthly_skips_nonexistent_days_and_honors_partial_month_bounds(self):
        actual = self.select(2023, init_type="monthly", init_days=(1, 15, 31),
                             start_date=(2023, 1, 15), end_date=(2023, 3, 10))
        pd.testing.assert_index_equal(actual, pd.DatetimeIndex(["2023-01-15", "2023-01-31", "2023-02-01", "2023-02-15", "2023-03-01"]))

    def test_leap_days_do_not_crash_weekly_or_monthly_selection(self):
        actual = self.select(2023, init_type="weekly", date_filter_year=2024, init_days=tuple(range(7)),
                             start_date=(2023, 2, 28), end_date=(2023, 3, 1))
        pd.testing.assert_index_equal(actual, pd.DatetimeIndex(["2023-02-28", "2023-03-01"]))
        leap = self.select(2024, init_type="monthly", init_days=(29,),
                           start_date=(2024, 2, 1), end_date=(2024, 2, 29))
        pd.testing.assert_index_equal(leap, pd.DatetimeIndex(["2024-02-29"]))
        nonleap = self.select(2023, init_type="monthly", init_days=(29,),
                              start_date=(2023, 2, 1), end_date=(2023, 2, 29))
        self.assertEqual(len(nonleap), 0)

    def test_wraparound_season_keeps_dates_within_the_target_file_year(self):
        actual = self.select(2023, init_type="monthly", init_days=(1,),
                             start_date=(2023, 11, 1), end_date=(2024, 2, 15))
        pd.testing.assert_index_equal(actual, pd.DatetimeIndex(["2023-01-01", "2023-02-01", "2023-11-01", "2023-12-01"]))
