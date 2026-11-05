"""Browser refreshes reconnect to a job without launching another benchmark."""

import threading
import time
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from ai_weather.benchmark_job import BenchmarkJob
from ai_weather.benchmark_registry import find_benchmark_job, register_benchmark_job
from ui.utils.benchmark_session import reconnect_benchmark, remember_benchmark


class BenchmarkReconnectTests(unittest.TestCase):
    def job(self, runner):
        return BenchmarkJob(runner, [{'mode': 'det', 'model_key': 'AIFS',
                                     'label': 'AIFS', 'config': {}}], 'settings')

    def test_fresh_session_attaches_same_job_without_starting_it(self):
        job = self.job(lambda *args, **kwargs: {})
        original, params = {}, {}
        remember_benchmark(original, params, job)
        refreshed = {}
        reconnect_benchmark(refreshed, params)
        self.assertIs(refreshed['benchmark_active_job'], job)
        self.assertFalse(job._started)
        self.assertEqual(refreshed['benchmark_job_token'], original['benchmark_job_token'])
        refreshed.pop('benchmark_active_job')
        reconnect_benchmark(refreshed, params)
        self.assertNotIn('benchmark_active_job', refreshed)

    def test_unknown_tokens_never_attach_another_sessions_job(self):
        job = self.job(lambda *args, **kwargs: {})
        token = register_benchmark_job(job)
        state = {}
        reconnect_benchmark(state, {'benchmark_job': token + '-invalid'})
        self.assertNotIn('benchmark_active_job', state)
        self.assertIn('benchmark_reconnect_error', state)
        self.assertIs(find_benchmark_job(token), job)
        self.assertIsNone(find_benchmark_job('unknown'))

    def test_existing_session_job_gets_a_refresh_token_without_restarting(self):
        job = self.job(lambda *args, **kwargs: {})
        state, params = {'benchmark_active_job': job}, {}
        reconnect_benchmark(state, params)
        self.assertIs(find_benchmark_job(params['benchmark_job']), job)
        self.assertFalse(job._started)
        token = params['benchmark_job']
        reconnect_benchmark(state, params)
        self.assertEqual(params['benchmark_job'], token)

    def test_completed_jobs_remain_available_for_refresh_until_expiration(self):
        job = self.job(lambda *args, **kwargs: {'ok': True})
        token = register_benchmark_job(job)
        job.start()
        self.assertTrue(job.done.wait(2))
        state = {}
        reconnect_benchmark(state, {'benchmark_job': token})
        self.assertEqual(state['benchmark_active_job'].snapshot()['status'], 'completed')
        with patch('ai_weather.benchmark_registry.time.monotonic', return_value=10**15):
            self.assertIsNone(find_benchmark_job(token))

    def test_refreshed_page_shows_progress_below_run_button_during_inventory_and_stops_same_job(self):
        import ui.pages.short_medium_benchmarking as page
        release = threading.Event()
        entered = threading.Event()
        calls = []

        def run(mode, key, config, *, progress, cancel_event):
            calls.append(key)
            progress('Processing year 2021')
            entered.set()
            while not cancel_event.wait(.02):
                if release.is_set():
                    break
            return {'ok': not cancel_event.is_set(), 'returncode': 0}

        job = self.job(run)
        token = register_benchmark_job(job)
        job.start()
        self.assertTrue(entered.wait(1))

        def inventory(*args):
            release.wait(10)
            return {'models': [], 'observation_years': [], 'loaded_at': 0}

        script = 'from ui.pages.short_medium_benchmarking import page\npage()'
        with patch.object(page, '_load_benchmark_inventory', side_effect=inventory):
            try:
                for inventory_ready in (False, True):
                    app = AppTest.from_string(script, default_timeout=10)
                    app.query_params.update(page='SM_Benchmarking', benchmark_job=token)
                    app.session_state['benchmark_last_run'] = {'ok': True}
                    if inventory_ready:
                        app.session_state['benchmark_inventory_cache'] = {
                            'CHIRPS': {'models': [], 'observation_years': [], 'loaded_at': time.time()}}
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertIs(app.session_state['benchmark_active_job'], job)
                    self.assertTrue(any('Processing year 2021' in item.value for item in app.code))
                    self.assertTrue(any('Benchmark running' in item.value for item in app.info))
                    self.assertFalse(any(item.label == 'Latest benchmark execution log'
                                         for item in app.expander))
                    self.assertTrue(any('Model 1 of 1: AIFS' in item.value for item in app.caption))
                    elements = list(app.main)
                    run_index = next(i for i, element in enumerate(elements)
                                     if element.type == 'button' and element.key == 'btn_run')
                    log_index = next(i for i, element in enumerate(elements)
                                     if element.type == 'code' and 'Processing year 2021' in element.value)
                    status_index = next(i for i, element in enumerate(elements)
                                        if element.type == 'info' and 'Benchmark running' in element.value)
                    self.assertLess(run_index, status_index)
                    self.assertLess(run_index, log_index)
                    self.assertTrue(app.button(key='btn_run').disabled)
                    cards = [item.value for item in app.markdown if '<div class="kpi-row">' in item.value]
                    self.assertEqual(len(cards), 1)
                    self.assertIn('>Running</div>', cards[0])
                    self.assertNotIn('>Idle</div>', cards[0])
                self.assertEqual(calls, ['AIFS'])
                app.button(key='btn_stop_benchmark').click().run()
                self.assertTrue(job.done.wait(2))
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(app.session_state['benchmark_last_run']['cancelled'])
                self.assertTrue(any(item.label == 'Latest benchmark execution log'
                                    for item in app.expander))
                self.assertNotIn('benchmark_active_job', app.session_state)
                app.run()  # Completed jobs must not reattach in a rerun loop.
                self.assertFalse(app.exception)
                refreshed = AppTest.from_string(script, default_timeout=10)
                refreshed.query_params.update(page='SM_Benchmarking', benchmark_job=token)
                refreshed.run()
                self.assertFalse(refreshed.exception)
                self.assertTrue(refreshed.session_state['benchmark_last_run']['cancelled'])
                self.assertNotIn('benchmark_active_job', refreshed.session_state)
                self.assertEqual(calls, ['AIFS'])
            finally:
                release.set()
                job.cancel()
                job.done.wait(2)
