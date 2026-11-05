"""Reject insufficient forecast coverage before scientific workers are launched."""

import ast
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps/benchmarking/ROMP'))
SOURCE = ROOT / 'run_momp.py'


def functions(*names, **namespace):
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class HorizonPreflightTests(unittest.TestCase):
    def cfg(self, folder, *, horizon=30, extent=21, dry_spell=7, years=(2021,)):
        return SimpleNamespace(model_list=('aifs_single',), model_dir_list=(folder,),
                               file_pattern_list=('{}.nc',), years=years,
                               start_date=(2021, 5, 1), end_date=(2022, 7, 31),
                               max_forecast_day=horizon, dry_extent=extent,
                               wet_spell=3, dry_spell=dry_spell, day_bins=((1, horizon),))

    def write_file(self, folder, year, days=45, *, hourly=False):
        # Deliberately noncanonical rainfall dimension order: transposing the
        # field must not be part of a metadata-only validation.
        ds = xr.Dataset({'tp': (('init_time', 'lat', 'lon', 'lead_time'),
                                np.zeros((1, 1, 1, days)))},
                        coords={'init_time': pd.to_datetime([f'{year}-05-01']),
                                'lat': [8.], 'lon': [39.],
                                'lead_time': xr.DataArray(np.arange(1, days + 1) * (24 if hourly else 1),
                                                         dims='lead_time',
                                                         attrs={'units': 'hours' if hourly else 'days'})})
        ds.to_netcdf(Path(folder) / f'{year}.nc', engine='netcdf4')

    def validate(self, cfg):
        return functions('validate_forecast_horizons')['validate_forecast_horizons'](cfg)

    def test_45_days_rejects_30_day_window_and_21_day_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_file(folder, 2021)
            with self.assertRaisesRegex(ValueError, 'aifs_single.*2021.*2021.nc.*day 45.*day 50'):
                self.validate(self.cfg(folder))

    def test_supported_horizons_and_disabled_dry_criterion_do_not_change_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_file(folder, 2021, hourly=True)
            for horizon, extent, dry_spell in ((30, 0, 7), (25, 21, 7), (15, 21, 7), (30, 21, 0)):
                with self.subTest(horizon=horizon, extent=extent, dry_spell=dry_spell):
                    cfg = self.cfg(folder, horizon=horizon, extent=extent, dry_spell=dry_spell)
                    expected = vars(cfg).copy()
                    self.validate(cfg)
                    self.assertEqual(vars(cfg), expected)

    def test_all_available_years_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_file(folder, 2021, days=50)
            self.write_file(folder, 2022, days=45)
            with self.assertRaisesRegex(ValueError, '2022.*day 45.*day 50'):
                self.validate(self.cfg(folder, years=(2021, 2022)))

    def test_validation_reads_coordinates_without_reading_rainfall(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_file(folder, 2021)
            original = NetCDF4ArrayWrapper.__getitem__
            seen = []

            def metadata_only(wrapper, key):
                seen.append(wrapper.variable_name)
                if wrapper.variable_name == 'tp':
                    raise AssertionError('Rainfall data was materialized during preflight')
                return original(wrapper, key)

            with patch.object(NetCDF4ArrayWrapper, '__getitem__', metadata_only):
                self.validate(self.cfg(folder, horizon=15))
            self.assertTrue(seen)
            self.assertNotIn('tp', seen)

    def test_failed_preflight_never_syncs_or_launches_driver(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'driver.py').touch()
            self.write_file(folder, 2021)
            cfg = self.cfg(folder)
            driver, sync = MagicMock(), MagicMock()
            preflight = functions('validate_forecast_horizons')['validate_forecast_horizons']
            ns = functions('run_single_model', _model_name=lambda *_: 'aifs_single',
                           MOMP_PKG_DIR=root, Path=Path, os=os, json=json,
                           config=SimpleNamespace(setup_earthdata_auth=MagicMock()),
                           patched_package_metadata=contextlib.nullcontext,
                           get_cfg=lambda **_: cfg, apply_base_paths=lambda _: None,
                           configure_model=lambda *_: None, apply_spec_to_cfg=MagicMock(),
                           apply_env_vars_to_cfg=lambda _: None, validate_runtime_config=lambda _: None,
                           ROMPValidator=lambda _: SimpleNamespace(validate=lambda: None),
                           validate_paths=lambda _: None, validate_forecast_horizons=preflight,
                           sync_params=sync, print_summary=lambda _: None, get_setting=lambda: None)
            with patch.dict(os.environ, {}, clear=True), \
                    patch.dict(sys.modules, {'momp.driver': SimpleNamespace(run_momp=driver)}):
                with self.assertRaisesRegex(ValueError, 'day 45.*day 50'):
                    ns['run_single_model']('det', 'AIFS', {})
            driver.assert_not_called()
            sync.assert_not_called()
