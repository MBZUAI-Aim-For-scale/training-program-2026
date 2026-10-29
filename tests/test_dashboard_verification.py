"""Keep editable verification windows, tolerances and forecast horizons consistent."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from ui.utils.verification_settings import parse_verification_settings

DAY_BINS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30))


class DashboardVerificationTests(unittest.TestCase):
    def render(self, state=None, forecast_window="1-15 days"):
        path = Path("ui/pages/short_medium_benchmarking.py")
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "render_verification_window_settings")
        st = MagicMock()
        st.session_state = dict(state or {})
        column = MagicMock()
        st.columns.return_value = [column, column]
        column.text_input.side_effect = lambda label, value, key: st.session_state.get(key, value)

        def number_input(label, *, min_value, max_value, value, key, help):
            current = st.session_state.get(key, value)
            self.assertGreaterEqual(current, min_value)
            self.assertLessEqual(current, max_value)
            return current

        column.number_input.side_effect = number_input
        ns = {"st": st, "config": SimpleNamespace(day_bins=DAY_BINS),
              "parse_verification_settings": parse_verification_settings}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        settings, errors = ns["render_verification_window_settings"]("model", forecast_window)
        return settings, errors, st

    def test_both_windows_raise_stale_horizon_and_pair_tolerances(self):
        settings, errors, st = self.render({"vwl_model_1": "((1, 15), (16, 30))",
                                           "max_forecast_day_model_1": 15})
        self.assertEqual(errors, [])
        self.assertEqual(settings["max_forecast_day"], 30)
        self.assertEqual(settings["tolerance_days_list"], "(3, 5)")
        self.assertEqual(st.session_state["max_forecast_day_model_1"], 30)

    def test_selected_window_defaults_and_valid_longer_horizon(self):
        for selection, expected, tolerance in (("1-15 days", 15, "(3,)"),
                                                ("16-30 days", 30, "(5,)")):
            with self.subTest(selection=selection):
                settings, errors, _ = self.render(forecast_window=selection)
                self.assertEqual(errors, [])
                self.assertEqual(settings["max_forecast_day"], expected)
                self.assertEqual(settings["tolerance_days_list"], tolerance)
        settings, _, _ = self.render({"max_forecast_day_model_1": 45})
        self.assertEqual(settings["max_forecast_day"], 45)

    def test_malformed_windows_and_mismatched_tolerances_show_errors(self):
        _, errors, _ = self.render({"vwl_model_1": "broken"})
        self.assertIn("Verification Window List", errors[0])
        _, errors, _ = self.render({"vwl_model_1": "((1, 15), (16, 30))",
                                    "tdl_model_((1, 15), (16, 30))": "(3,)"})
        self.assertIn("one nonnegative integer per verification window", errors[0])

    def test_parser_accepts_single_pair_and_rejects_unusable_windows(self):
        self.assertEqual(parse_verification_settings("(16, 30)", "5"), (((16, 30),), (5,), 30))
        for text in ("()", "((30, 16),)", "((1, 61),)", "((1, 15), (1, 15))",
                     "((1, 14),)", "((True, 15),)", "((1, 15, 30),)"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_verification_settings(text, day_bins=DAY_BINS)
