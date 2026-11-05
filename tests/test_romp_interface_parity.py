"""Both entry points must merge the same inputs and reject invalid overrides."""

import ast
import contextlib
import json
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


SOURCE = Path('run_momp.py')


def functions(*names, **namespace):
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class InterfaceParityTests(unittest.TestCase):
    def namespace(self):
        catalog = {'CHIRPS': {'dir': '/chirps', 'file_pattern': '{}.nc', 'var': 'precip'},
                   'ENACTS': {'dir': '/enacts', 'file_pattern': ['rain_{}.nc', '{}.nc'], 'var': 'rain'}}
        return functions('_clean_str', '_meta_get', '_obs_meta', '_iso_to_tuple',
                         '_apply_forecast_window', 'apply_spec_to_cfg', 'apply_env_vars_to_cfg',
                         config=SimpleNamespace(BENCHMARK_OBS_CATALOG=catalog), os=os,
                         ast=ast, date=date, resolve_path=lambda path: path)

    def cfg(self):
        return SimpleNamespace(obs='CHIRPS', obs_dir='/chirps', obs_var='precip',
                               obs_file_pattern=('{}.nc',), ref_model='climatology',
                               ref_model_dir='/chirps', ref_model_var='precip',
                               ref_model_file_pattern=('{}.nc',))

    def test_spec_and_dashboard_ground_truth_and_temporal_settings_match(self):
        ns = self.namespace()
        spec_cfg, dashboard_cfg = self.cfg(), self.cfg()
        spec = {'ground_truth': 'ENACTS', 'forecast_window': '16-30',
                'shared': {'dry_threshold': 2.5},
                'models': [{'exec_key': 'model', 'eval_start': '2023-05-01',
                            'eval_end': '2024-07-31', 'init_type': 'monthly',
                            'initialization_days': [1, 15], 'date_filter_year': 2020,
                            'verification_window_list': [[16, 30]],
                            'tolerance_days_list': [4], 'max_forecast_day': 30,
                            'run_years_concurrently': False, 'ensemble_forecast': True}]}
        ns['apply_spec_to_cfg'](spec_cfg, spec, 'model')
        environment = {'ROMP_GROUND_TRUTH': 'ENACTS', 'ROMP_FORECAST_WINDOW': '16-30',
                       'ROMP_DRY_THRESHOLD': '2.5',
                       'ROMP_EVAL_START': '2023-05-01', 'ROMP_EVAL_END': '2024-07-31',
                       'ROMP_INIT_TYPE': 'monthly', 'ROMP_INIT_DAYS': '1,15',
                       'ROMP_DATE_FILTER_YEAR': '2020', 'ROMP_VERIFICATION_WINDOW_LIST': '((16,30),)',
                       'ROMP_TOLERANCE_DAYS_LIST': '(4,)', 'ROMP_MAX_FORECAST_DAY': '30',
                       'ROMP_RUN_YEARS_CONCURRENTLY': '0', 'ROMP_ENSEMBLE_FORECAST': '1'}
        with patch.dict(os.environ, environment, clear=True):
            ns['apply_env_vars_to_cfg'](dashboard_cfg)
        self.assertEqual(vars(spec_cfg), vars(dashboard_cfg))
        self.assertEqual(spec_cfg.ref_model_dir, '/enacts')
        self.assertEqual(spec_cfg.ref_model_var, 'rain')
        self.assertEqual(spec_cfg.obs_file_pattern, ('rain_{}.nc', '{}.nc'))
        self.assertEqual(spec_cfg.dry_threshold, 2.5)

    def test_spec_with_other_models_never_silently_uses_default_settings(self):
        ns = self.namespace()
        with self.assertRaisesRegex(ValueError, 'model|match'):
            ns['apply_spec_to_cfg'](self.cfg(),
                {'models': [{'exec_key': 'other', 'baseline_start_year': 1980}]}, 'requested')

    def test_shared_only_spec_remains_supported(self):
        cfg = self.cfg()
        self.namespace()['apply_spec_to_cfg'](cfg, {'shared': {'dry_threshold': 2.5}}, 'model')
        self.assertEqual(cfg.dry_threshold, 2.5)

    def test_invalid_environment_overrides_fail_instead_of_reusing_defaults(self):
        ns = self.namespace()
        for name, value in {'ROMP_EVAL_START': '2023-02-30', 'ROMP_WET_THRESHOLD': 'bad',
                            'ROMP_DRY_SPELL': '3.5', 'ROMP_VERIFICATION_WINDOW_LIST': 'bad',
                            'ROMP_TOLERANCE_DAYS_LIST': '()', 'ROMP_MAX_FORECAST_DAY': 'bad',
                            'ROMP_INIT_DAYS': 'mon', 'ROMP_ENSEMBLE_FORECAST': 'maybe',
                            'ROMP_FORECAST_WINDOW': 'nonsense', 'ROMP_GROUND_TRUTH': 'UNKNOWN'}.items():
            with self.subTest(name=name), patch.dict(os.environ, {name: value}, clear=True):
                with self.assertRaises(ValueError):
                    ns['apply_env_vars_to_cfg'](self.cfg())

    def test_invalid_spec_types_fail_instead_of_coercing_or_using_defaults(self):
        ns = self.namespace()
        for override in ({'init_days': ['Monday']}, {'ensemble_forecast': 'false'},
                         {'eval_start': '2023-02-30'}, {'run_years_concurrently': 'false'}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                ns['apply_spec_to_cfg'](self.cfg(), {'models': [{'exec_key': 'model', **override}]}, 'model')

    def test_invalid_or_missing_spec_never_runs_driver_with_unrequested_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'driver.py').touch()
            bad = root / 'bad.json'
            bad.write_text('{invalid json')
            cfg = self.cfg()
            driver = MagicMock()
            ns = functions('run_single_model', _model_name=lambda *_: 'model',
                           MOMP_PKG_DIR=root, Path=Path, os=os, json=json,
                           config=SimpleNamespace(setup_earthdata_auth=MagicMock()),
                           patched_package_metadata=contextlib.nullcontext,
                           get_cfg=lambda **_: cfg, apply_base_paths=lambda _: None,
                           configure_model=lambda *_: None, apply_spec_to_cfg=MagicMock(),
                           apply_env_vars_to_cfg=lambda _: None, validate_runtime_config=lambda _: None,
                           ROMPValidator=lambda _: SimpleNamespace(validate=lambda: None),
                           validate_paths=lambda _: None, validate_forecast_horizons=lambda _: None,
                           sync_params=lambda _: None,
                           print_summary=lambda _: None, get_setting=lambda: None)
            for spec_path in (bad, root / 'missing.json'):
                with self.subTest(path=spec_path), patch.dict(os.environ, {'ROMP_SPEC_PATH': str(spec_path)}, clear=True), \
                        patch.dict(sys.modules, {'momp.driver': SimpleNamespace(run_momp=driver)}):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        ns['run_single_model']('det', 'model', {})
            driver.assert_not_called()


class ExecutionIsolationTests(unittest.TestCase):
    def test_in_process_models_start_from_same_base_configuration(self):
        state = {'cfg': SimpleNamespace(wet_threshold=20)}
        seen = []

        def get_cfg(**kwargs):
            return SimpleNamespace(**vars(state['cfg']))

        def set_cfg(cfg):
            state['cfg'] = SimpleNamespace(**vars(cfg))

        def run_model(mode, key, meta):
            cfg = get_cfg()
            seen.append(cfg.wet_threshold)
            cfg.wet_threshold = 99
            set_cfg(cfg)

        ns = functions('execute_momp_benchmark', os=os, sys=sys, Path=Path, __file__=str(SOURCE.resolve()),
                       PROJECT_ROOT=Path.cwd(), select_runtime_python=lambda _: os.path.abspath(sys.executable),
                       check_runtime_python=lambda *_: 'dependencies ready',
                       select_models=lambda *_: [('first', {}), ('second', {})],
                       _model_name=lambda meta, key: key, run_single_model=run_model, get_cfg=get_cfg)
        with patch.dict(sys.modules, {'momp.lib.loader': SimpleNamespace(set_cfg=set_cfg)}):
            self.assertEqual(ns['execute_momp_benchmark']('det', in_process=True), 0)
        self.assertEqual(seen, [20, 20])

    def test_runner_honors_configured_output_roots(self):
        tree = ast.parse(SOURCE.read_text())
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id in {'ROMP_OUT_DIR', 'ROMP_FIG_DIR'}
                               for target in node.targets)]
        namespace = {'Path': Path, 'os': os, 'PROJECT_ROOT': Path('/project'),
                     'config': SimpleNamespace(ROMP_OUT_DIR='/default/results', ROMP_FIG_DIR='/default/figures')}
        with patch.dict(os.environ, {'ROMP_OUT_DIR': '/custom/results', 'ROMP_FIG_DIR': '/custom/figures'}, clear=True):
            exec(compile(ast.Module(body=assignments, type_ignores=[]), str(SOURCE), 'exec'), namespace)
        self.assertEqual(namespace['ROMP_OUT_DIR'], Path('/custom/results'))
        self.assertEqual(namespace['ROMP_FIG_DIR'], Path('/custom/figures'))


class CanonicalEntryTests(unittest.TestCase):
    def test_notebook_executor_delegates_to_canonical_runner(self):
        source = Path('notebooks/run_momp.py')
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'execute_momp_benchmark')
        canonical = SimpleNamespace(execute_momp_benchmark=MagicMock(return_value=3))
        scope = {'os': os, 'sys': sys, 'Path': Path, '__file__': str(source.resolve()),
                 'PROJECT_ROOT': Path.cwd(), 'select_runtime_python': lambda _: os.path.abspath(sys.executable),
                 'check_runtime_python': lambda *_: 'dependencies ready', '_canonical_runner': lambda: canonical,
                 'select_models': MagicMock(side_effect=AssertionError('Stale duplicated runner executed'))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
        self.assertEqual(scope['execute_momp_benchmark']('prob', 'model', True), 3)
        canonical.execute_momp_benchmark.assert_called_once_with('prob', 'model', True)

    def test_metadata_patch_does_not_fabricate_versions_for_missing_dependencies(self):
        import importlib.metadata
        from importlib.metadata import PackageNotFoundError
        ns = functions('patched_package_metadata', contextlib=contextlib,
                       importlib=__import__('importlib'), PackageNotFoundError=PackageNotFoundError)
        with ns['patched_package_metadata']():
            self.assertEqual(importlib.metadata.version('momp'), '0.0.1')
            with self.assertRaises(PackageNotFoundError):
                importlib.metadata.version('romp-audit-nonexistent-dependency')


class NotebookUtilityTests(unittest.TestCase):
    def test_notebook_utility_uses_canonical_subprocess_even_in_correct_kernel(self):
        source = Path('notebooks/utils.py')
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'execute_momp_benchmark')
        worker = MagicMock(return_value={'ok': True})
        scope = {'os': os, 'sys': sys, 'PROJECT_ROOT': Path.cwd(),
                 'select_runtime_python': lambda _: os.path.abspath(sys.executable),
                 'run_momp_benchmark': worker, 'check_runtime_python': lambda *_: 'ready'}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
        scope['execute_momp_benchmark']('det')
        worker.assert_called_once_with(mode='det', stream=True)
        worker.return_value = {'ok': False, 'stderr': 'invalid input'}
        with self.assertRaisesRegex(RuntimeError, 'invalid input'):
            scope['execute_momp_benchmark']('det')


class SubprocessBridgeParityTests(unittest.TestCase):
    def test_notebook_and_dashboard_forward_the_same_explicit_settings(self):
        from typing import Any, Dict, Optional
        captured = []
        settings = {'eval_start': '2023-05-01', 'eval_end': '2024-07-31',
                    'area_mask_file': '', 'onset_threshold_file': '/threshold.csv',
                    'baseline_forecast': 'climatology', 'wet_day_threshold': 30.0,
                    'minimum_wet_day_rainfall': 1.0, 'wet_spell_length': 3,
                    'dry_spell_limit': 7, 'dry_spell_search_extension': 21,
                    'dry_threshold': 2.5,
                    'initialization_days': [1, 15], 'init_type': 'monthly',
                    'date_filter_year': 2020, 'verification_window_list': ((1, 15),),
                    'tolerance_days_list': (3,), 'max_forecast_day': 15}
        with tempfile.TemporaryDirectory() as directory:
            for filename in ('ui/pages/short_medium_benchmarking.py', 'notebooks/utils.py'):
                source = Path(filename)
                node = next(node for node in ast.parse(source.read_text()).body
                            if isinstance(node, ast.FunctionDef) and node.name == 'run_momp_benchmark')
                env_seen = []

                def spawn(argv, env, *args, **kwargs):
                    env_seen.append(env)
                    return 0, False

                def run(argv, **kwargs):
                    env_seen.append(kwargs['env'])
                    return SimpleNamespace(returncode=0, stdout='', stderr='')

                scope = {'os': os, 'sys': sys, 'Path': Path, 'Any': Any, 'Dict': Dict, 'Optional': Optional,
                         'PROJECT_ROOT': Path(directory), 'ROMP_DEMO_OUT_DIR': Path(directory),
                         '_locate_run_momp': lambda: Path('/project/run_momp.py'),
                         'select_runtime_python': lambda _: sys.executable,
                         'check_runtime_python': lambda *_: 'dependencies ready',
                         '_spawn_no_fork': spawn, 'subprocess': SimpleNamespace(run=run),
                         'clear_caches': lambda: None}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
                with patch.dict(os.environ, {}, clear=True):
                    if filename.startswith('ui/'):
                        scope['run_momp_benchmark']('det', run_config=settings, progress=lambda _: None)
                    else:
                        scope['run_momp_benchmark']('det', run_config=settings)
                self.assertTrue(env_seen)
                captured.append({key: value for key, value in env_seen[-1].items() if key.startswith('ROMP_')})
        self.assertEqual(*captured)
        self.assertEqual(captured[0]['ROMP_DRY_THRESHOLD'], '2.5')


class InputValidationParityTests(unittest.TestCase):
    def test_fixed_observation_filename_accepted_by_loader_also_passes_preflight(self):
        ns = functions('validate_paths', DATA_DIR='/data',
                       _path_exists=lambda _: True, _list_nc=lambda _: ['observations.nc'],
                       fnmatch=__import__('fnmatch'))
        cfg = SimpleNamespace(obs_dir='/obs', obs_file_pattern=('observations.nc',),
                              start_date=(2023, 5, 1), end_date=(2024, 7, 31),
                              ref_model_dir='/obs', ref_model_file_pattern=('observations.nc',))
        ns['validate_paths'](cfg)

    def test_spec_never_truncates_fractional_day_counts(self):
        ns = InterfaceParityTests().namespace()
        with self.assertRaisesRegex(ValueError, 'integer'):
            ns['apply_spec_to_cfg'](InterfaceParityTests().cfg(),
                                   {'shared': {'wet_spell_length': 3.9}}, 'model')
