"""Fresh interpreters must not opt spawn-based jobs into gRPC fork hooks."""

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GrpcProcessStartupTests(unittest.TestCase):
    def environment(self):
        env = dict(os.environ, DEMO_DATA_SOURCE='local', PYTHONPATH=str(ROOT))
        env.pop('GRPC_ENABLE_FORK_SUPPORT', None)
        env.pop('GRPC_POLL_STRATEGY', None)
        return env

    def test_startup_disables_fork_hooks_and_preserves_explicit_settings(self):
        script = '''
import json, os
from ai_weather import config
from momp.stats import parallel
print(json.dumps([os.getenv('GRPC_ENABLE_FORK_SUPPORT'), os.getenv('GRPC_POLL_STRATEGY')]))
'''
        for overrides, expected in (({}, ['0', None]),
                                    ({'GRPC_ENABLE_FORK_SUPPORT': '1', 'GRPC_POLL_STRATEGY': 'epoll1'},
                                     ['1', 'epoll1'])):
            with self.subTest(overrides=overrides):
                env = {**self.environment(), **overrides,
                       'MPLCONFIGDIR': str(ROOT / '.mplcache')}
                env['PYTHONPATH'] += os.pathsep + str(ROOT / 'apps/benchmarking/ROMP')
                result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout.splitlines()[-1]), expected)

    @unittest.skipUnless(importlib.util.find_spec('grpc'), 'optional gRPC runtime is absent')
    def test_spawn_worker_completes_with_a_grpc_channel_in_the_parent(self):
        script = '''
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from ai_weather import config
import grpc

def worker():
    import os
    return os.environ['GRPC_ENABLE_FORK_SUPPORT']

if __name__ == '__main__':
    with grpc.insecure_channel('localhost:1'):
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn')) as pool:
            assert pool.submit(worker).result(timeout=30) == '0'
    print('worker completed')
'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'grpc_spawn_check.py'
            path.write_text(script)
            result = subprocess.run([sys.executable, str(path)], cwd=ROOT,
                                    env=self.environment(), capture_output=True, text=True,
                                    timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('worker completed', result.stdout)
        self.assertNotIn('fork_posix.cc', result.stderr)
        self.assertNotIn('absl::InitializeLog()', result.stderr)
