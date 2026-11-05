"""Onset Analysis must use the same resolved source and credentials as scoring."""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch


SOURCE = Path('ui/pages/short_medium_benchmarking.py')


def functions(*names, **namespace):
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class OnsetInputSourceTests(unittest.TestCase):
    def test_bucket_reader_produces_real_onset_plot_without_a_local_observation_directory(self):
        import contextlib
        import io
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import xarray as xr

        sys.path.insert(0, str(Path.cwd() / 'apps/benchmarking/ROMP'))
        # The real isolated worker clears its argv before importing ROMP too.
        with patch.object(sys, 'argv', [sys.argv[0]]):
            from momp.app.onset_time_series import obs_onset_analysis
        from momp.utils import storage

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for year in (2020, 2021):
                dates = pd.date_range(f'{year}-05-01', f'{year}-05-31')
                rainfall = np.where(dates.day >= 10, 10., 0.)[:, None, None]
                data = xr.DataArray(rainfall, dims=('time', 'lat', 'lon'),
                                    coords={'time': dates, 'lat': [8.5], 'lon': [39.25]},
                                    name='RAINFALL', attrs={'units': 'mm/day'})
                data.to_dataset().to_netcdf(root / f'{year}.nc')

            class Bucket:
                def exists(self, path):
                    return (root / Path(path).name).is_file()

                def open(self, path, mode):
                    return (root / Path(path).name).open(mode)

            settings = dict(obs_dir='gs://observations/CHIRPS/Ethiopia',
                            obs_file_pattern=('{}.nc',), obs_var='RAINFALL', obs_unit_cvt=1,
                            start_year_clim=2020, end_year_clim=2021,
                            start_date=(2020, 5, 1), end_date=(2021, 5, 31), fallback_date=None,
                            wet_threshold=20., wet_init=1., wet_spell=3,
                            dry_threshold=1., dry_spell=7, dry_extent=0, mok=None,
                            nc_mask='', region='Ethiopia', thresh_file=None,
                            dir_fig=str(root), extend_end_day=0)
            with patch.object(storage, 'config', SimpleNamespace()), \
                    patch.object(storage, 'gcs_fs', return_value=Bucket()), \
                    patch.object(plt, 'show'), contextlib.redirect_stdout(io.StringIO()):
                try:
                    _, onset = obs_onset_analysis(2021, lat_select=8.5, lon_select=39.25, **settings)
                    self.assertEqual(pd.Timestamp(onset.item()), pd.Timestamp('2021-05-10'))
                    figures = list(root.glob('onset_time_series_*.png'))
                    self.assertEqual(len(figures), 1)
                    self.assertTrue(figures[0].read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                finally:
                    plt.close('all')

    def config(self):
        return SimpleNamespace(CHIRPS_025_DIR='/app/data/external/CHIRPS_regridded_025',
                               CHIRPS_DIR='gs://observations/CHIRPS/Ethiopia',
                               OBS_DIR='gs://observations/CHIRPS/Ethiopia', obs_var='RAINFALL')

    def test_resolved_bucket_source_takes_precedence_over_missing_nominal_folder(self):
        ns = functions('_raw_onset_obs_dir', config=self.config(), Path=Path)
        self.assertEqual(ns['_raw_onset_obs_dir'](), 'gs://observations/CHIRPS/Ethiopia')

    def test_worker_kwargs_keep_remote_source_without_copying_archive_or_using_stale_cache(self):
        discover = MagicMock(return_value=(('{}.nc',), [2020, 2021]))
        prepare = MagicMock(side_effect=AssertionError('Observation archive must not be staged'))
        ns = functions('_raw_onset_obs_dir', '_get_onset_base_kwargs', config=self.config(),
                       Path=Path, ROMP_DEMO_FIG_DIR=Path('/figures'),
                       _onset_config_kwargs=lambda: {'obs_dir': '/stale/cache', 'region': 'Ethiopia'},
                       _discover_dataset_structure=discover, _prepare_local_obs_dir=prepare)
        settings, source = ns['_get_onset_base_kwargs']()
        self.assertEqual(source, 'gs://observations/CHIRPS/Ethiopia')
        self.assertEqual(settings['obs_dir'], source)
        self.assertEqual(settings['ref_model_dir'], source)
        self.assertEqual(settings['obs_file_pattern'], ('{}.nc',))
        discover.assert_called_once_with(source)
        prepare.assert_not_called()

    def test_isolated_worker_receives_container_bucket_and_authentication_environment(self):
        import os
        import pickle

        startup = {'PATH': '/opt/envs/app/bin:/usr/bin', 'DEMO_BUCKET': 'observations',
                   'DEMO_DATA_SOURCE': 'bucket',
                   'GOOGLE_APPLICATION_CREDENTIALS': '/mounted/credentials.json'}
        captured = []

        def spawn(arguments, env, cwd, timeout, on_line):
            captured.append(dict(env))
            payload = pickle.loads(Path(arguments[-2]).read_bytes())
            self.assertEqual(payload['kwargs']['obs_dir'], 'gs://observations/CHIRPS/Ethiopia')
            destination = Path(arguments[-1])
            destination.mkdir()
            (destination / 'figure.png').write_bytes(b'png-result')
            return 0, False

        with tempfile.TemporaryDirectory() as directory:
            ns = functions('_analyze_onset_streamlit', Path=Path, List=list, os=os,
                           PROJECT_ROOT=Path(directory), _STARTUP_ENV=startup,
                           _get_onset_base_kwargs=lambda: ({'obs_dir': 'gs://observations/CHIRPS/Ethiopia'},
                                                         'gs://observations/CHIRPS/Ethiopia'),
                           _snap_to_grid=lambda lat, lon, source: (lat, lon),
                           _ONSET_WORKER_SOURCE='worker', select_runtime_python=lambda _: '/opt/envs/app/bin/python',
                           check_runtime_python=lambda *args: None, _spawn_no_fork=spawn)
            result = ns['_analyze_onset_streamlit'](2021, 8.5, 39.25)
        self.assertIsNone(result['error'])
        self.assertEqual(result['pngs'], [b'png-result'])
        for key, value in startup.items():
            self.assertEqual(captured[0][key], value)
