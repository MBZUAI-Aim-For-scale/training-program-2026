"""Input-readiness UI keeps year, date and source controls synchronized."""

import copy
import unittest
from datetime import date
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from ai_weather.blending import blending_outputs


class BlendingInputUITests(unittest.TestCase):
    def app(self):
        return AppTest.from_string(
            "from ui.pages.ai_onset_blending import page\npage()",
            default_timeout=30,
        )

    @staticmethod
    def inspect_and_wait(app):
        app.button(key="blending_inspect_inputs").click().run()
        if "blending_input_job" in app.session_state:
            assert app.session_state["blending_input_job"].done.wait(2)
            app.run()

    @staticmethod
    def report(settings, ok=True, common_dates=None):
        return {
            "ok": ok,
            "errors": [] if ok else ["single: filename year differs from decoded dates"],
            "records": [
                {
                    "source": "single",
                    "file": "/tmp/forecast.nc",
                    "filename_year": settings["year"],
                    "actual_years": str(settings["year"] if ok else 2024),
                    "first_date": "2024-06-13",
                    "last_date": "2024-06-17",
                    "valid": ok,
                    "error": "" if ok else "Year mismatch",
                }
            ],
            "common_dates": common_dates or [],
            "signature": "mock-signature",
            "fingerprints": [],
            "_checked": {
                key: settings[key]
                for key in ("year", "issue_date", "aifs_nc_folder", "aifs_ens_nc_folder")
            },
        }

    @staticmethod
    def matches(settings, report):
        return bool(report) and report.get("_checked") == {
            key: settings[key]
            for key in ("year", "issue_date", "aifs_nc_folder", "aifs_ens_nc_folder")
        }

    def test_inspection_is_explicit_and_valid_check_enables_run(self):
        import ui.pages.ai_onset_blending as page

        app = self.app()
        with (
            patch.object(page, "inspect_blending_inputs", side_effect=self.report) as inspect,
            patch.object(page, "input_check_matches", side_effect=self.matches),
        ):
            app.run()
            inspect.assert_not_called()
            self.assertTrue(app.button(key="blending_run").disabled)
            self.inspect_and_wait(app)
            inspect.assert_called_once()
            self.assertFalse(app.exception)
            self.assertFalse(app.button(key="blending_run").disabled)
            self.assertEqual(
                sum("filename_year" in item.value.columns for item in app.dataframe), 1
            )
            self.assertTrue(any("match the selected year" in item.value for item in app.success))
            current = app.date_input(key="blending_date").value
            changed = date(current.year, current.month, current.day + 1)
            app.date_input(key="blending_date").set_value(changed).run()
            self.assertTrue(app.button(key="blending_run").disabled)
            self.assertTrue(any("stale" in item.value for item in app.warning))

    def test_preview_and_resumed_application_bypass_inspection(self):
        app = self.app().run()
        self.assertTrue(app.button(key="blending_run").disabled)
        app.checkbox(key="blending_dry_run").check().run()
        self.assertFalse(app.button(key="blending_run").disabled)
        app.checkbox(key="blending_dry_run").uncheck()
        app.number_input(key="blending_start").set_value(6).run()
        self.assertFalse(app.button(key="blending_run").disabled)

    def test_invalid_files_are_visible_and_execution_stays_disabled(self):
        import ui.pages.ai_onset_blending as page

        app = self.app()
        with (
            patch.object(
                page,
                "inspect_blending_inputs",
                side_effect=lambda settings: self.report(settings, ok=False),
            ),
            patch.object(page, "input_check_matches", side_effect=self.matches),
        ):
            app.run()
            self.inspect_and_wait(app)
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key="blending_run").disabled)
            self.assertTrue(any("filename year" in item.value for item in app.error))
            self.assertEqual(app.dataframe[0].value.iloc[0]["actual_years"], "2024")

    def test_available_date_synchronizes_controls_without_changing_sources(self):
        import ui.pages.ai_onset_blending as page

        for custom_output in (None, "/tmp/my custom output"):
            with self.subTest(custom_output=custom_output):
                captured = []

                def inspect(settings, captured=captured):
                    captured.append(copy.deepcopy(settings))
                    return self.report(
                        settings, ok=False, common_dates=["2024-06-13", "2024-06-17"]
                    )

                app = self.app()
                with (
                    patch.object(page, "inspect_blending_inputs", side_effect=inspect),
                    patch.object(page, "input_check_matches", side_effect=self.matches),
                ):
                    app.run()
                    old_year = app.number_input(key="blending_year").value
                    model = app.selectbox(key="blending_model").value
                    companion = app.selectbox(key="blending_companion").value
                    old_context = f"{model}_{companion}_{old_year}"
                    app.text_input(key="blending_single_folder").set_value("/tmp/custom single")
                    app.text_input(key="blending_ens_folder").set_value("/tmp/custom ensemble")
                    for section in ("single", "ensemble"):
                        app.text_input(
                            key=f"blending_spec_{old_context}_{section}_input_file_regex"
                        ).set_value(rf"^{old_year}_custom\.nc$")
                    app.number_input(
                        key=f"blending_spec_{old_context}_single_options_window"
                    ).set_value(5)
                    app.text_input(
                        key=f"blending_spec_{old_context}_maps_country_boundary"
                    ).set_value("/tmp/custom-country.geojson")
                    if custom_output:
                        app.text_input(key="blending_work_dir").set_value(custom_output)
                    app.run()
                    self.inspect_and_wait(app)
                    app.button(key="blending_use_available_date").click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(app.number_input(key="blending_year").value, 2024)
                    self.assertEqual(app.date_input(key="blending_date").value, date(2024, 6, 13))
                    self.assertEqual(
                        app.number_input(
                            key=f"blending_spec_{model}_{companion}_2024_single_options_window"
                        ).value,
                        5,
                    )
                    self.assertEqual(
                        app.text_input(
                            key=f"blending_spec_{model}_{companion}_2024_maps_country_boundary"
                        ).value,
                        "/tmp/custom-country.geojson",
                    )
                    self.assertEqual(
                        app.text_input(key="blending_single_folder").value, "/tmp/custom single"
                    )
                    self.assertEqual(
                        app.text_input(key="blending_ens_folder").value, "/tmp/custom ensemble"
                    )
                    for section in ("single", "ensemble"):
                        self.assertEqual(
                            app.text_input(
                                key=f"blending_spec_{model}_{companion}_2024_{section}_input_file_regex"
                            ).value,
                            r"^2024_custom\.nc$",
                        )
                    expected_output = custom_output or str(
                        blending_outputs({"year": 2024, "issue_date": "2024-06-13"})["root"]
                    )
                    self.assertEqual(
                        app.text_input(key="blending_work_dir").value, expected_output
                    )
                    self.assertTrue(app.button(key="blending_run").disabled)
                    self.assertIsNone(app.session_state["blending_input_check"])
                    self.assertTrue(
                        any("Inspect forecast inputs again" in item.value for item in app.info)
                    )
                    self.assertEqual(len(captured), 1)
                    self.assertEqual(captured[0]["year"], old_year)

    def test_inspection_exception_is_presented(self):
        import ui.pages.ai_onset_blending as page

        app = self.app()
        with (
            patch.object(
                page, "inspect_blending_inputs", side_effect=OSError("cannot read source")
            ),
            patch.object(page, "input_check_matches", return_value=False),
        ):
            app.run()
            self.inspect_and_wait(app)
        self.assertFalse(app.exception)
        self.assertTrue(any("cannot read source" in item.value for item in app.error))
        self.assertTrue(app.button(key="blending_run").disabled)


if __name__ == "__main__":
    unittest.main()
