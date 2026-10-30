"""Cancellation messages must not disguise actual failures or unverified cleanup."""

import unittest

from ui.utils.benchmark_messages import cancellation_warning, execution_status


class BenchmarkMessageTests(unittest.TestCase):
    def test_requested_cancellation_is_not_labelled_as_failure(self):
        result = {"cancelled": True, "returncode": 130, "ok": False}
        self.assertEqual(execution_status(result), "Cancelled by you")
        self.assertEqual(cancellation_warning({"runs": [result]}), "")

    def test_legacy_permission_trace_has_honest_cleanup_status(self):
        result = {"returncode": 1, "ok": False, "stderr": "PermissionError: Operation not permitted"}
        self.assertIn("cleanup not confirmed", execution_status(result, cancelled=True))
        self.assertIn("Some workers may still be running", cancellation_warning({"runs": [result]}))
        self.assertEqual(execution_status(result), "Failed — exit code 1")

    def test_completed_models_remain_completed_when_later_run_is_cancelled(self):
        self.assertEqual(execution_status({"ok": True, "returncode": 0}, cancelled=True), "Completed")
