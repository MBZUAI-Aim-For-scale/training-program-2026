"""Forecast polling must not block interactions or lose live work on settings changes."""
import ast
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from apps.forecasting import runs


class ForecastResponsivenessTests(unittest.TestCase):
    def state(self, **changes):
        return dict(sig=('original',), started=False, paths={}, queue=[], running=None,
                    job=None, failed={}, plan={}, download=None, **changes)

    def test_download_returns_immediately_polls_and_does_not_launch_next_model_early(self):
        entered, release = threading.Event(), threading.Event()

        def download(*args):
            entered.set()
            release.wait(3)
            return Path('/tmp/saved.nc')

        plan = {'saved': (runs.DOWNLOAD, ('unique-test.nc', 123), None),
                'gpu': (runs.RUN, '', 'request')}
        state = self.state()
        with patch.object(runs.store, 'download', side_effect=download) as worker, \
             patch.object(runs.jobs, 'launch', return_value='gpu-job') as launch:
            try:
                runs.start(state, plan)
                runs.tick(state)
                self.assertTrue(entered.wait(1))
                for _ in range(5):
                    runs.tick(state)
                launch.assert_not_called()
                self.assertEqual(state['running'], 'saved')
                self.assertIs(runs._download('saved', 'unique-test.nc'), state['download'])
                worker.assert_called_once()
                runs.start(state, plan)
                self.assertEqual(state['queue'], ['gpu'])
                release.set()
                state['download'].result(timeout=2)
                runs.tick(state)
                self.assertEqual(state['paths']['saved'], '/tmp/saved.nc')
                launch.assert_called_once_with('request')
            finally:
                release.set()

    def test_independent_downloads_can_run_concurrently(self):
        entered = {name: threading.Event() for name in ('first', 'second')}
        release = threading.Event()

        def download(model, name):
            entered[model].set()
            release.wait(3)
            return Path('/tmp') / name

        with patch.object(runs.store, 'download', side_effect=download):
            futures = [runs._download(name, f'{name}-parallel.nc') for name in entered]
            try:
                for event in entered.values():
                    self.assertTrue(event.wait(1))
            finally:
                release.set()
                for future in futures:
                    future.result(timeout=2)

    def test_busy_state_survives_control_change_then_resets_when_finished(self):
        state = self.state()
        state.update(started=True, running='gpu', job='job-id')
        session = {'runs': state}
        with patch.object(runs.st, 'session_state', session):
            self.assertIs(runs.state(('changed',)), state)
            state['running'] = None
            self.assertEqual(runs.state(('changed',))['sig'], ('changed',))

    def test_failed_download_unblocks_queue_and_records_error(self):
        from concurrent.futures import Future
        future = Future()
        future.set_exception(OSError('offline'))
        state = self.state()
        state.update(running='saved', download=future)
        runs.tick(state)
        self.assertFalse(runs.busy(state))
        self.assertIn('offline', state['failed']['saved'])

    def test_inactive_result_tabs_skip_all_result_work(self):
        source = Path('ui/pages/ai_forecasting.py')
        tree = ast.parse(source.read_text())
        render = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render')
        guards = [n for n in render.body if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Attribute) and n.test.attr == 'open'
                  and isinstance(n.test.value, ast.Subscript)
                  and isinstance(n.test.value.value, ast.Name)
                  and n.test.value.value.id == 'result_tabs']
        self.assertEqual(len(guards), 4)
        ns = {'result_tabs': [SimpleNamespace(open=False)] * 4}
        for guard in guards:
            # Closed tabs must not even access undefined result/data/UI functions.
            exec(compile(ast.Module(body=[guard], type_ignores=[]), str(source), 'exec'), ns)

    def test_slow_discovery_survives_reruns_and_never_reuses_obsolete_plan(self):
        from ai_weather.background_task import BackgroundTask
        source = Path('ui/pages/ai_forecasting.py')
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_discovery_result')
        state = {}
        ns = {'ss': state, 'BackgroundTask': BackgroundTask}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), ns)
        entered, release = threading.Event(), threading.Event()
        calls = []

        def discover(value):
            calls.append(value)
            entered.set()
            release.wait(3)
            return {'plan': value}

        discover_result = ns['_discovery_result']
        try:
            started = time.monotonic()
            self.assertEqual(discover_result('plan', ('old',), discover, 'old'), (False, None))
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(entered.wait(1))
            original = state['plan']['task']
            for _ in range(5):
                self.assertEqual(discover_result('plan', ('old',), discover, 'old'), (False, None))
                self.assertIs(state['plan']['task'], original)
            self.assertEqual(discover_result('plan', ('new',), discover, 'new'), (False, None))
            self.assertEqual(calls, ['old'])
            release.set()
            self.assertTrue(original.done.wait(1))
            discover_result('plan', ('new',), discover, 'new')
            self.assertTrue(state['plan']['task'].done.wait(1))
            self.assertEqual(discover_result('plan', ('new',), discover, 'new'),
                             (True, {'plan': 'new'}))
            self.assertEqual(calls, ['old', 'new'])
        finally:
            release.set()

    def test_onset_movie_launch_is_an_isolated_nonblocking_process(self):
        with patch('subprocess.Popen') as spawn:
            job = runs.launch_onset_movies([])
            try:
                self.assertIs(job['process'], spawn.return_value)
                command = spawn.call_args.args[0]
                self.assertIn('onset_view.make_movie', command[2])
                self.assertEqual(spawn.call_args.kwargs['env']['MPLBACKEND'], 'Agg')
            finally:
                job['folder'].cleanup()
