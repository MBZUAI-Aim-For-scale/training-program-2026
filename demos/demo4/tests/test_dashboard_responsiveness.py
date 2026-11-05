"""Menu routing remains cheap even when unrelated weather services are unavailable."""

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from ai_weather.background_task import BackgroundTask


class BackgroundTaskTests(unittest.TestCase):
    def test_nonblocking_duplicate_safe_and_independent(self):
        release = threading.Event()
        entered = threading.Event()
        seen = []

        def worker(settings):
            seen.append(settings['value'])
            entered.set()
            release.wait(3)
            return settings['value']

        settings = {'value': 1}
        task = BackgroundTask(worker, settings)
        settings['value'] = 2
        try:
            task.start().start()
            self.assertTrue(entered.wait(1))
            self.assertFalse(task.snapshot()['done'])
            independent = BackgroundTask(lambda: 42).start()
            self.assertTrue(independent.done.wait(1))
            self.assertEqual(independent.snapshot()['result'], 42)
        finally:
            release.set()
        self.assertTrue(task.done.wait(1))
        self.assertEqual(seen, [1])
        self.assertEqual(task.snapshot()['result'], 1)

    def test_exception_is_visible_and_task_finishes(self):
        def fail():
            raise ValueError('service unavailable')
        task = BackgroundTask(fail).start()
        self.assertTrue(task.done.wait(1))
        self.assertEqual(task.snapshot()['status'], 'failed')
        self.assertIn('service unavailable', task.snapshot()['error'])


class MenuIsolationTests(unittest.TestCase):
    def test_repeated_navigation_only_imports_and_runs_selected_menu(self):
        import importlib
        original = importlib.import_module
        calls = []
        rendered = []

        def load(name, *args, **kwargs):
            if name.startswith('ui.pages.'):
                calls.append(name)
                return SimpleNamespace(page=lambda: rendered.append(name))
            return original(name, *args, **kwargs)

        app = AppTest.from_file(str(Path('ui/app.py').resolve()), default_timeout=10)
        routes = ['Home', 'AI Forecast', 'SM_Benchmarking', 'Onset Blending', 'Onset Message']
        modules = ['home', 'ai_forecasting', 'short_medium_benchmarking', 'ai_onset_blending', 'onset_message']
        with patch('importlib.import_module', side_effect=load):
            app.run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(calls, ['ui.pages.home'])
            for _ in range(2):
                for route, module in zip(routes, modules):
                    calls.clear()
                    rendered.clear()
                    app.button(key=f'primary_nav_{route}').click().run()
                    self.assertEqual(list(app.exception), [])
                    self.assertTrue(rendered)
                    self.assertEqual(set(rendered), {f'ui.pages.{module}'})
                    # A navigation click may run the old shell before st.rerun;
                    # it must not import/render the old or any unrelated page.
                    self.assertEqual(set(calls), {f'ui.pages.{module}'})


class OutputSerializationTests(unittest.TestCase):
    def test_same_output_waits_but_independent_output_runs_and_waiter_can_cancel(self):
        from ai_weather.benchmark_job import BenchmarkJob
        release = threading.Event()
        entered = threading.Event()
        calls = []

        def runner(mode, key, settings, **kwargs):
            calls.append(key)
            if key == 'first':
                entered.set()
                release.wait(3)
            return {'ok': True}

        def job(key, resource):
            return BenchmarkJob(runner, [{'mode': 'det', 'model_key': key,
                                         'label': key, 'config': {}}], key,
                                resource_key=resource)

        first = job('first', 'shared').start()
        try:
            self.assertTrue(entered.wait(1))
            first.start()  # Calling start twice must not launch duplicate writers.
            waiter = job('waiter', 'shared').start()
            other = job('other', 'independent').start()
            self.assertTrue(other.done.wait(1))
            self.assertFalse(waiter.done.is_set())
            waiter.cancel()
            self.assertTrue(waiter.done.wait(1))
            self.assertEqual(waiter.snapshot()['status'], 'cancelled')
            self.assertEqual(calls, ['first', 'other'])
        finally:
            release.set()
        self.assertTrue(first.done.wait(1))


class NavigationDuringWorkTests(unittest.TestCase):
    def test_real_shell_navigation_preserves_pending_message_job(self):
        import ui.pages.onset_message as message
        release = threading.Event()
        entered = threading.Event()

        def connect(token):
            entered.set()
            release.wait(5)
            return {'token': 'test', 'districts': ['District']}

        app = AppTest.from_file(str(Path('ui/app.py').resolve()), default_timeout=5)
        with patch.object(message, 'SHARED_API_KEY', 'test'), \
                patch.object(message, '_connect', side_effect=connect) as connect_mock:
            try:
                app.run()
                app.button(key='primary_nav_Onset Message').click().run()
                self.assertTrue(entered.wait(1))
                task = app.session_state['onset_connection_job']
                for _ in range(2):
                    app.button(key='primary_nav_Home').click().run()
                    self.assertFalse(app.exception)
                    self.assertTrue(any('Task running in background' in i.value for i in app.info))
                    app.button(key='primary_nav_Onset Message').click().run()
                    self.assertFalse(app.exception)
                    self.assertIs(app.session_state['onset_connection_job'], task)
                connect_mock.assert_called_once()
            finally:
                release.set()
            self.assertTrue(task.done.wait(1))
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['onset_districts'], ['District'])
            self.assertFalse(any('Task running' in i.value for i in app.info))
