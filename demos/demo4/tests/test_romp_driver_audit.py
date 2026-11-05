"""Analytic regressions for driver validation, case construction and parallel parity."""

import argparse
import ast
import copy
import os
import runpy
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

ROOT = Path('apps/benchmarking/ROMP/momp')


def functions(relative, names, **namespace):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, 'exec'), namespace)
    return namespace


class DriverAuditTests(unittest.TestCase):
    def cfg(self, **changes):
        return dict(start_date=(2024, 5, 1), end_date=(2024, 7, 31),
                    start_year_clim=2000, end_year_clim=2023,
                    model_list=('model',), verification_window_list=((1, 15),),
                    tolerance_days_list=(3,), day_bins=((1, 5), (6, 10), (11, 15)),
                    max_forecast_day=15, dry_extent=21, dry_spell=7, thresh_file=None,
                    probabilistic=False, **changes)

    def validator(self, values):
        namespace = runpy.run_path(str(ROOT / 'lib/assertion.py'))
        return namespace['ROMPValidator'](values), namespace['ROMPConfigError']

    def test_empty_or_malformed_configs_fail_with_configuration_error(self):
        for updates in ({'model_list': ()}, {'verification_window_list': ()},
                        {'verification_window_list': ((1,),)}, {'day_bins': ((1, 5, 9),)},
                        {'tolerance_days_list': (-1,)}, {'day_bins': ((0, 5),)},
                        {'wet_spell': 0}, {'onset_percentage_threshold': 1.1},
                        {'day_bins': ((1, 5), (5, 10))}, {'day_bins': ((1, 20),)}):
            with self.subTest(updates=updates):
                values = self.cfg()
                values.update(updates)
                validator, error = self.validator(values)
                with self.assertRaises(error):
                    validator.validate()

    def test_validator_is_reusable_after_config_is_corrected(self):
        values = self.cfg()
        values['model_list'] = ('invalid model',)
        validator, error = self.validator(values)
        with self.assertRaises(error):
            validator.validate()
        validator.cfg.model_list = ('valid_model',)
        self.assertTrue(validator.validate())

    def test_reference_override_uses_requested_model(self):
        cfg = SimpleNamespace(ref_model='other', probabilistic=True,
                              model_list=('forecast',), layout=('model',))
        ns = functions('lib/control.py', {'ref_cfg_layout'}, copy=copy,
                       iter_list=lambda values: [values['model_list']])
        result, layout = ns['ref_cfg_layout'](cfg, ref_model='climatology')
        self.assertEqual(result.model_list, ('climatology',))
        self.assertFalse(result.probabilistic)
        self.assertEqual(cfg.ref_model, 'other')

    def test_reference_case_accepts_documented_dictionary_setting(self):
        namespace = runpy.run_path(str(ROOT / 'lib/convention.py'))
        case = namespace['Case'](model='climatology')
        settings = {'ref_model_dir': '/reference', 'ref_model_var': 'rain',
                    'ref_model_file_pattern': '{}.nc', 'ref_model_unit_cvt': 1.0}
        ns = functions('lib/control.py', {'ref_model_case'}, copy=copy, asdict=asdict,
                       Setting=namespace['Setting'], is_dataclass=lambda value: False)
        result, merged = ns['ref_model_case'](case, settings)
        self.assertEqual(result.model_dir, '/reference')
        self.assertEqual(merged['model_var'], 'rain')
        self.assertIsNone(case.model_var)

    def test_window_parser_rejects_empty_and_nonpair_windows(self):
        ns = functions('lib/parser.py', {'parse_window_list'}, argparse=argparse)
        for value in ('', '1', '1,2,3'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                ns['parse_window_list'](value)

    def test_parallel_climatology_matches_year_weighted_mean_with_nan_and_uneven_chunks(self):
        values = {2000: [100., np.nan], 2001: [101., np.nan], 2002: [200., 250.]}

        def dataset(**kwargs):
            return xr.DataArray([values[year] for year in kwargs['years_clim']],
                                dims=('year', 'point'), coords={'year': kwargs['years_clim']})

        def means(**kwargs):
            return np.round(dataset(**kwargs).mean('year', skipna=True))

        ns = functions('stats/parallel.py', {'parallel_climatological_onset',
                       'parallel_climatological_onset_dataset', 'func_wrapper_climatology'},
                       os=os, np=np, xr=xr, partial=partial, ThreadPoolExecutor=ThreadPoolExecutor,
                       compute_climatological_onset=means,
                       compute_climatological_onset_dataset=dataset)
        with patch.object(os, 'cpu_count', return_value=2):
            result = ns['parallel_climatological_onset'](years_clim=[2000, 2001, 2002])
        np.testing.assert_equal(result.values, [134., 250.])

    def test_empty_parallel_requests_fail_before_creating_executors(self):
        ns = functions('stats/parallel.py', {'parallel_climatological_onset_dataset'},
                       os=os, np=np, xr=xr, ThreadPoolExecutor=MagicMock())
        with self.assertRaisesRegex(ValueError, 'years_clim'):
            ns['parallel_climatological_onset_dataset'](years_clim=[])
        ns['ThreadPoolExecutor'].assert_not_called()

    def test_explicit_driver_cfg_derives_matching_setting_and_validates_before_stages(self):
        namespace = runpy.run_path(str(ROOT / 'lib/convention.py'))
        validator, _ = self.validator(self.cfg())
        cfg = SimpleNamespace(**self.cfg(), dir_out='/explicit-output')
        stage = MagicMock()
        ns = functions('driver.py', {'run_momp'},
                       get_cfg=MagicMock(), get_setting=lambda: SimpleNamespace(dir_out='/stale'),
                       init_dataclass=lambda dc, values: dc(**{k: v for k, v in values.items() if k in dc.__annotations__}),
                       Setting=namespace['Setting'], ROMPValidator=type(validator),
                       print_momp_banner=MagicMock(), logger=MagicMock(),
                       skill_score_in_bins=stage, spatial_far_mr_mae_map=MagicMock())
        ns['run_momp'](cfg=cfg)
        self.assertEqual(stage.call_args.kwargs['setting'].dir_out, '/explicit-output')
        stage.reset_mock()
        cfg.model_list = ()
        with self.assertRaises(Exception):
            ns['run_momp'](cfg=cfg)
        stage.assert_not_called()

    def test_explicit_loader_args_rebuild_cached_cfg_and_setting(self):
        ns = functions('lib/loader.py', {'get_cfg'},
                       _cfg={'wet_threshold': 20}, _setting='stale',
                       build_cfg=lambda args: {'wet_threshold': int(args[1])},
                       SimpleNamespace=SimpleNamespace)
        result = ns['get_cfg'](['--wet_threshold', '35'], validate=False)
        self.assertEqual(result.wet_threshold, 35)
        self.assertIsNone(ns['_setting'])

    def test_parallel_climatology_keeps_valid_years_when_one_chunk_has_no_data(self):
        def dataset(**kwargs):
            years = list(kwargs['years_clim'])
            if years == [2000]:
                raise ValueError('No years were successfully processed')
            return xr.DataArray([[year] for year in years], dims=('year', 'point'),
                                coords={'year': years})
        ns = functions('stats/parallel.py', {'parallel_climatological_onset_dataset',
                       'func_wrapper_climatology'}, os=os, np=np, xr=xr, partial=partial,
                       ThreadPoolExecutor=ThreadPoolExecutor,
                       compute_climatological_onset_dataset=dataset)
        with patch.object(os, 'cpu_count', return_value=3):
            actual = ns['parallel_climatological_onset_dataset'](years_clim=[2000, 2001, 2002])
        self.assertEqual(actual.year.values.tolist(), [2001, 2002])
