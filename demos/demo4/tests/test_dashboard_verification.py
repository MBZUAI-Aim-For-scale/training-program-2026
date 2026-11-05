"""Advanced controls serialize only supported, internally consistent settings."""

import ast
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from ui.utils.verification_settings import (
    initialization_matches_schedule,
    parse_verification_settings,
    temporal_settings_errors,
)

DAY_BINS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30))


class DashboardVerificationTests(unittest.TestCase):
    def render(self, state=None, forecast_window="1-15 days", function_name="render_verification_window_settings", config=None):
        path = Path("ui/pages/short_medium_benchmarking.py")
        function = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == function_name)
        st = MagicMock()
        st.session_state = dict(state or {})
        column = MagicMock()
        st.columns.side_effect = lambda n: [column] * n
        st.selectbox.side_effect = lambda label, options, index=0, **kw: st.session_state.get(kw["key"], options[index])
        st.multiselect.side_effect = lambda label, options, default, **kw: st.session_state.get(kw["key"], default)
        def number(label, *, value, **kw):
            return st.session_state.get(kw["key"], value)
        st.number_input.side_effect = number
        column.number_input.side_effect = number
        ns = {"st": st, "config": config or SimpleNamespace(day_bins=DAY_BINS),
              "parse_verification_settings": parse_verification_settings}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        result = ns[function_name]("model", forecast_window) if function_name == "render_verification_window_settings" else ns[function_name]("model")
        return result, st

    def test_both_windows_ignore_stale_manual_horizon_and_tuple_fields(self):
        (settings, errors), st = self.render({"verification_windows_model_0": "Both windows",
                                             "max_forecast_day_model_1": 15, "vwl_model_1": "broken"})
        self.assertEqual(errors, [])
        self.assertEqual(settings["max_forecast_day"], 30)
        self.assertEqual(settings["verification_window_list"], "((1, 15), (16, 30))")
        self.assertEqual(settings["tolerance_days_list"], "(3, 5)")
        self.assertTrue(st.number_input.call_args.kwargs["disabled"])

    def test_each_window_has_an_automatic_horizon(self):
        for selection, expected, tolerance in (("1-15 days", 15, "(3,)"), ("16-30 days", 30, "(5,)")):
            with self.subTest(selection=selection):
                (settings, errors), _ = self.render(forecast_window=selection)
                self.assertEqual(errors, [])
                self.assertEqual(settings["max_forecast_day"], expected)
                self.assertEqual(settings["tolerance_days_list"], tolerance)

    def test_custom_tolerance_and_invalid_config_bins(self):
        (settings, errors), _ = self.render({"verification_tolerance_model_1_15": 4})
        self.assertEqual(settings["tolerance_days_list"], "(4,)")
        self.assertEqual(errors, [])
        (_, errors), _ = self.render(config=SimpleNamespace(day_bins=((1, 5),)))
        self.assertIn("do not cover", errors[0])

    def test_modes_have_separate_valid_day_controls(self):
        for mode, days in (("weekly", [0, 3]), ("monthly", [1, 15]), ("daily", list(range(7)))):
            with self.subTest(mode=mode):
                settings, st = self.render({"initialization_schedule_model": mode}, function_name="render_initialization_settings")
                self.assertEqual(settings["initialization_days"], days)
                self.assertEqual(settings["init_type"], mode)
                self.assertEqual(st.number_input.call_args.kwargs["disabled"], mode != "weekly")
                if mode != "daily":
                    expected = list(range(7)) if mode == "weekly" else list(range(1, 32))
                    self.assertEqual(st.multiselect.call_args.args[1], expected)

    def test_parser_still_accepts_legacy_payload_and_rejects_unusable_windows(self):
        self.assertEqual(parse_verification_settings("(16, 30)", "5"), (((16, 30),), (5,), 30))
        for text in ("()", "((30, 16),)", "((1, 61),)", "((1, 15), (1, 15))", "((1, 14),)"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_verification_settings(text, day_bins=DAY_BINS)

    def test_temporal_preflight_detects_empty_invalid_and_missing_selections(self):
        base = {"eval_start": date(2020, 5, 1), "eval_end": date(2024, 7, 31),
                "baseline_start_year": 2007, "baseline_end_year": 2024,
                "init_type": "weekly", "initialization_days": [0, 3], "date_filter_year": 2024}
        years = list(range(1998, 2025))
        self.assertEqual(temporal_settings_errors(base, years), [])
        for patch, message in (({"initialization_days": []}, "weekday"),
                               ({"initialization_days": [15]}, "weekday"),
                               ({"date_filter_year": None}, "reference"),
                               ({"baseline_start_year": 2025}, "Baseline"),
                               ({"eval_end": date(2026, 7, 31)}, "missing"),
                               ({"eval_end": base["eval_start"]}, "after"),
                               ({"init_type": "monthly", "initialization_days": [31],
                                 "eval_start": date(2020, 2, 1), "eval_end": date(2024, 2, 28)}, "do not occur")):
            with self.subTest(patch=patch):
                self.assertTrue(any(message in e for e in temporal_settings_errors({**base, **patch}, years)))
        self.assertTrue(temporal_settings_errors({**base, "probabilistic": True, "baseline_start_year": 2024}, years))

    def test_available_initializations_use_reference_calendar_and_season(self):
        settings = {"eval_start": date(2003, 5, 1), "eval_end": date(2023, 7, 31),
                    "init_type": "weekly", "initialization_days": [0, 3], "date_filter_year": 2024}
        self.assertTrue(initialization_matches_schedule((2003, 5, 2), settings))
        self.assertFalse(initialization_matches_schedule((2003, 5, 1), settings))
        self.assertFalse(initialization_matches_schedule((2003, 8, 2), settings))
        settings.update(init_type="monthly", initialization_days=[1, 15])
        self.assertTrue(initialization_matches_schedule((2003, 5, 15), settings))
        self.assertFalse(initialization_matches_schedule((2003, 5, 2), settings))
