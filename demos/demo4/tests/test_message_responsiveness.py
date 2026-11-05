"""Message API work must not block reruns or repeat on widget interaction."""

import threading
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import ui.pages.onset_message as message


class MessageResponsivenessTests(unittest.TestCase):
    def app(self, connected=False):
        app = AppTest.from_string(
            "from ui.pages.onset_message import page\npage()", default_timeout=5,
        )
        if connected:
            app.session_state["onset_token"] = "test-token"
            app.session_state["onset_districts"] = ["First", "Second"]
        return app

    def button(self, app, label):
        return next(button for button in app.button if button.label == label)

    def test_connection_is_background_and_districts_survive_repeated_reruns(self):
        release = threading.Event()
        entered = threading.Event()

        def connect(token):
            entered.set()
            release.wait(5)
            return {"token": "test-token", "districts": ["First", "Second"]}

        app = self.app()
        with patch.object(message, "SHARED_API_KEY", "test-key"), \
                patch.object(message, "_connect", side_effect=connect) as connect_mock:
            try:
                app.run()
                self.assertTrue(entered.wait(1))
                job = app.session_state["onset_connection_job"]
                for _ in range(3):
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertTrue(any("Task running" in item.value for item in app.info))
                connect_mock.assert_called_once()
            finally:
                release.set()
            self.assertTrue(job.done.wait(1))
            app.run()
            for value in (0.45, 0.50, 0.55):
                app.slider(key="t1").set_value(value).run()
                self.assertFalse(app.exception)
            connect_mock.assert_called_once()
            self.assertEqual(app.session_state["onset_districts"], ["First", "Second"])

    def test_forecast_runs_in_background_and_prevents_duplicate_submission(self):
        release = threading.Event()
        entered = threading.Event()

        def fetch(*args):
            entered.set()
            release.wait(5)
            return {"results": [{"district": "First", "chance": "High"}], "errors": []}

        app = self.app(connected=True)
        with patch.object(message, "_fetch_forecasts", side_effect=fetch) as fetch_mock, \
                patch.object(message, "get_districts") as districts_mock:
            try:
                app.run()
                self.button(app, "Generate Forecast").click().run()
                self.assertTrue(entered.wait(1))
                job = app.session_state["onset_forecast_job"]
                for value in (0.45, 0.50, 0.55):
                    app.slider(key="t1").set_value(value).run()
                    self.assertFalse(app.exception)
                    self.assertTrue(self.button(app, "Generate Forecast").disabled)
                    self.assertTrue(self.button(app, "Reset Session").disabled)
                    self.assertTrue(any("Task running" in item.value for item in app.info))
                fetch_mock.assert_called_once()
                districts_mock.assert_not_called()
            finally:
                release.set()
            self.assertTrue(job.done.wait(1))
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["forecast_results"][0]["district"], "First")
            self.assertTrue(app.success)
            self.assertFalse(self.button(app, "Generate Forecast").disabled)
            self.button(app, "Reset Session").click()
            with patch.object(message, "SHARED_API_KEY", ""):
                app.run()
            self.assertNotIn("forecast_results", app.session_state)

    def test_partial_api_failures_keep_successful_results_and_request_order(self):
        with patch.object(message, "get_forecast", side_effect=[
                [{"district": "First"}], RuntimeError("backend unavailable")]) as fetch:
            result = message._fetch_forecasts("token", ["First", "Second"], False, [.4], "season")
        self.assertEqual(result["results"], [{"district": "First"}])
        self.assertIn("Second: RuntimeError: backend unavailable", result["errors"])
        self.assertEqual([call.args[1] for call in fetch.call_args_list], ["First", "Second"])

    def test_connection_failure_is_visible_and_does_not_auto_retry(self):
        app = self.app()
        with patch.object(message, "SHARED_API_KEY", "test-key"), \
                patch.object(message, "_connect", side_effect=RuntimeError("unauthorized")) as connect:
            app.run()
            job = app.session_state["onset_connection_job"] if "onset_connection_job" in app.session_state else None
            if job:
                self.assertTrue(job.done.wait(1))
            app.run()
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any("unauthorized" in item.value for item in app.error))
            connect.assert_called_once()
