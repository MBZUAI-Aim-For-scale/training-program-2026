"""Slow inspection and inactive result tabs do not block unrelated controls."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class BlendingResponsivenessTests(unittest.TestCase):
    def test_slow_inspection_remains_responsive_and_prevents_duplicates(self):
        import ui.pages.ai_onset_blending as page

        release = threading.Event()
        entered = threading.Event()

        def inspect(settings):
            entered.set()
            release.wait(5)
            return {"ok": True, "errors": [], "records": [], "common_dates": []}

        app = AppTest.from_string(
            'from ui.pages.ai_onset_blending import page\npage()', default_timeout=3
        )
        try:
            with patch.object(page, 'inspect_blending_inputs', side_effect=inspect) as check:
                app.run()
                app.button(key='blending_inspect_inputs').click().run()
                self.assertTrue(entered.wait(1))
                self.assertFalse(app.exception)
                self.assertTrue(app.button(key='blending_inspect_inputs').disabled)
                self.assertTrue(app.button(key='blending_run').disabled)
                self.assertTrue(any('Task running' in i.value for i in app.info))
                for _ in range(2):
                    app.checkbox(key='blending_dry_run').set_value(True).run()
                    self.assertFalse(app.exception)
                    self.assertTrue(app.button(key='blending_run').disabled)
                self.assertEqual(check.call_count, 1)
                task = app.session_state['blending_input_job']
                release.set()
                self.assertTrue(task.done.wait(1))
                app.run()
                self.assertNotIn('blending_input_job', app.session_state)
                self.assertFalse(any('Task running' in i.value for i in app.info))
        finally:
            release.set()

    def test_only_selected_result_tab_loads_assets(self):
        import ui.pages.ai_onset_blending as page

        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / 'summary.csv'
            summary.write_text('id,week1\ndistrict,0.2\n')
            image = Path(directory) / 'map.png'
            image.write_bytes(b'not loaded while its tab is closed')
            outputs = {'root': Path(directory), 'summary': summary,
                       'predictions': Path(directory) / 'missing.csv', 'maps': [image]}
            app = AppTest.from_string(
                'from ui.pages.ai_onset_blending import render_results\n'
                'render_results({"issue_date": "2026-06-15"})', default_timeout=3
            )
            with patch.object(page, 'blending_outputs', return_value=outputs):
                with patch.object(Path, 'read_bytes', side_effect=AssertionError('inactive asset read')):
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertEqual(len(app.dataframe), 1)
                    self.assertFalse(app.get('download_button'))
                    self.assertFalse(app.get('imgs'))
                app.session_state['blending_result_tabs'] = 'Data & Export'
                with patch.object(page.pd, 'read_csv', side_effect=AssertionError('inactive table read')):
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertEqual(len(app.get('download_button')), 2)
                    self.assertFalse(app.dataframe)
