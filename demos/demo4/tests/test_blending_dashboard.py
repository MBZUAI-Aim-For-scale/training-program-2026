"""Dashboard controls share the CLI and retain completed job results."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from ai_weather.blending import blending_outputs, build_blending_command, run_blending


class BlendingAdapterTests(unittest.TestCase):
    def test_options_remain_separate_command_arguments(self):
        command = build_blending_command(
            {
                "year": 2026,
                "issue_date": "2026-06-15",
                "work_dir": "/tmp/path with spaces",
                "dry_run": True,
            }
        )
        self.assertEqual(command[command.index("--work_dir") + 1], "/tmp/path with spaces")
        self.assertIn("--dry_run", command)
        with self.assertRaises(ValueError):
            build_blending_command({"unexpected": "value"})

    def test_runner_streams_logs_and_reports_failure(self):
        settings = {"year": 2026, "issue_date": "2026-06-15"}

        def spawn(*args):
            args[4]("step failed")
            return 1, False

        with patch("ai_weather.blending.spawn_process", side_effect=spawn):
            progress = []
            result = run_blending("blending", "aifs", settings, progress=progress.append)
        self.assertFalse(result["ok"])
        self.assertEqual(progress, ["step failed"])
        self.assertEqual(result["stdout"], "step failed")

    def test_inventory_is_scoped_to_selected_issue_date(self):
        outputs = blending_outputs(
            {"year": 2027, "issue_date": "2027-06-15", "work_dir": "/tmp/output"}
        )
        self.assertEqual(outputs["summary"].name, "blend_output_summary_20270615.csv")
        self.assertTrue(all("20270615" in path.name for path in outputs["maps"]))


class BlendingDashboardTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "ui/pages/ai_onset_blending.py"),
            default_timeout=30,
        ).run()

    def test_page_loads_and_rejects_inverted_stage_range(self):
        app = self.app()
        self.assertEqual(list(app.exception), [])
        app.number_input(key="blending_start").set_value(7)
        app.number_input(key="blending_end").set_value(3)
        app.run()
        self.assertTrue(app.button(key="blending_run").disabled)
        self.assertTrue(app.error)

    def test_completed_preview_remains_visible_after_rerun(self):
        import ui.pages.ai_onset_blending as page

        def runner(*args, **kwargs):
            kwargs["progress"]("[INFO] --- Step 1/8: Preview ---")
            return {"ok": True, "returncode": 0, "stdout": "preview commands", "dry_run": True}

        # A wrapper keeps the page module patchable across Streamlit reruns.
        app = AppTest.from_string(
            "from ui.pages.ai_onset_blending import page\npage()", default_timeout=30
        )
        with patch.object(page, "run_blending", side_effect=runner):
            app.run()
            app.checkbox(key="blending_dry_run").check().run()
            app.button(key="blending_run").click().run()
            for _ in range(5):
                if app.success:
                    break
                time.sleep(0.05)
                app.run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any("preview completed" in item.value for item in app.success))
        app.run()
        self.assertTrue(app.success)

    def test_saved_probability_table_and_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.app()
            stamp = app.date_input(key="blending_date").value.strftime("%Y%m%d")
            Path(directory, f"blend_output_summary_{stamp}.csv").write_text(
                "id,time,year,week1,week2,week3,week4,later\n"
                "district,2026-06-15,2026,0.2,0.2,0.2,0.2,0.2\n"
            )
            app.text_input(key="blending_work_dir").set_value(directory).run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(len(app.dataframe), 1)
            self.assertEqual(app.dataframe[0].value.iloc[0]["id"], "district")
            self.assertEqual(len(app.get("download_button")), 0)
            app.session_state["blending_result_tabs"] = "Data & Export"
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("download_button")), 1)


if __name__ == "__main__":
    unittest.main()
