"""Operational settings are editable and passed from the page to its job runner."""

import copy
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from ai_weather import config
from ai_weather.blending_config import runtime_spec_defaults


class BlendingParameterUITests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "ui/pages/ai_onset_blending.py"),
            default_timeout=30,
        ).run()

    def key(self, section, path, model=None, companion=None, year=None):
        model = model or config.DEFAULT_MODEL_SINGLE
        companion = companion or (config.DEFAULT_MODEL_ENS if model == "aifs" else "aifs")
        return f"blending_spec_{model}_{companion}_{year or config.YEAR}_{section}_{path}"

    def test_config_defaults_cover_scientific_and_map_settings(self):
        app = self.app()
        self.assertFalse(app.exception)
        defaults = runtime_spec_defaults(config.DEFAULT_MODEL_SINGLE, config.YEAR)
        self.assertEqual(
            app.number_input(key=self.key("single", "options_window")).value,
            defaults["single"]["options"]["window"],
        )
        self.assertEqual(
            app.text_input(key=self.key("ensemble", "mok_file")).value,
            defaults["ensemble"]["mok"]["file"],
        )
        self.assertEqual(
            app.text_input(key=self.key("single", "thresholds_file")).value,
            defaults["single"]["thresholds"]["file"],
        )
        self.assertEqual(
            app.number_input(
                key=self.key("climatology", "climatologies_clim_train_year_min")
            ).value,
            defaults["climatology"]["climatologies"]["clim"]["train_year_min"],
        )
        self.assertEqual(
            app.text_input(key=self.key("maps", "country_boundary")).value,
            str(config.ETH_GEOJSON_PATH),
        )
        self.assertTrue(app.number_input(key=self.key("single", "input_workers")).disabled)
        self.assertTrue(app.checkbox(key=self.key("single", "input_parallel")).disabled)
        self.assertTrue(
            app.text_input(key=self.key("combine", "options_default_prob_col")).disabled
        )
        self.assertTrue(app.number_input(key=self.key("combine", "climatology_max_day")).disabled)
        self.assertEqual(
            app.text_input(key="blending_single_folder").value,
            defaults["single"]["input"]["nc_folder"],
        )
        self.assertEqual(
            app.text_input(key="blending_gt").value,
            str(config.ONSET_FORECAST_IMD_CLIM_SPEC["input"]["gt_path"]),
        )

    def test_invalid_mapping_yaml_disables_execution(self):
        app = self.app()
        app.text_area(key=self.key("single", "dimensions_rename")).set_value("invalid: [").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertTrue(app.button(key="blending_run").disabled)

    def test_year_and_model_refresh_dependent_defaults(self):
        app = self.app()
        app.text_input(key="blending_coef_dir").set_value("/tmp/custom coefficients").run()
        app.number_input(key="blending_year").set_value(2024).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.date_input(key="blending_date").value.year, 2024)
        self.assertIn(
            "2024", app.text_input(key=self.key("single", "input_file_regex", year=2024)).value
        )
        self.assertTrue(app.text_input(key="blending_work_dir").value.endswith("/2024"))
        app.selectbox(key="blending_model").select("gencast").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="blending_companion").value, "aifs")
        self.assertEqual(
            app.text_input(key="blending_single_folder").value,
            config.ONSET_FORECAST_GENCAST_SPEC["input"]["nc_folder"],
        )
        self.assertEqual(app.text_input(key="blending_coef_dir").value, "/tmp/custom coefficients")

    def test_invalid_lead_range_disables_execution(self):
        app = self.app()
        app.number_input(key=self.key("single", "options_min_day")).set_value(50).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button(key="blending_run").disabled)
        self.assertTrue(any("lead day" in item.value for item in app.error))

    def test_changed_settings_reach_background_job(self):
        import ui.pages.ai_onset_blending as page

        captured = []

        def runner(mode, model, settings, **kwargs):
            captured.append(copy.deepcopy(settings))
            return {"ok": True, "returncode": 0, "stdout": "preview", "dry_run": True}

        app = AppTest.from_string(
            "from ui.pages.ai_onset_blending import page\npage()", default_timeout=30
        )
        with patch.object(page, "run_blending", side_effect=runner):
            app.run()
            app.number_input(key=self.key("single", "options_window")).set_value(5)
            app.number_input(
                key=self.key("ensemble", "options_onset_definition_wet_day_min_mm")
            ).set_value(2.5)
            app.text_input(key=self.key("maps", "district_boundary")).set_value(
                "/tmp/custom.geojson"
            )
            app.text_input(key="blending_input").set_value("/tmp/custom.pkl")
            app.text_input(key="blending_region").set_value("Selected region")
            app.checkbox(key="blending_dry_run").check().run()
            app.button(key="blending_run").click().run()
            for _ in range(10):
                if captured:
                    break
                time.sleep(0.05)
                app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(captured), 1)
        settings = captured[0]
        self.assertEqual(settings["runtime_overrides"]["single"]["options"]["window"], 5)
        self.assertEqual(
            settings["runtime_overrides"]["ensemble"]["options"]["onset_definition"][
                "wet_day_min_mm"
            ],
            2.5,
        )
        self.assertEqual(
            settings["runtime_overrides"]["maps"]["district_boundary"], "/tmp/custom.geojson"
        )
        self.assertEqual(settings["blend_input"], "/tmp/custom.pkl")
        self.assertEqual(settings["region"], "Selected region")
        self.assertEqual(
            settings["runtime_overrides"]["single"]["input"]["nc_folder"],
            settings["aifs_nc_folder"],
        )


if __name__ == "__main__":
    unittest.main()
