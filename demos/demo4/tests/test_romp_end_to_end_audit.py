"""Real NetCDF -> CLI/dashboard subprocess -> scientific output parity.

All inputs, boundaries and outputs are local synthetic fixtures. These tests run
actual onset/scoring/plotting code; they do not replace scientific functions.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE_ROOT))
PYTHON = str(ROOT / '.venv' / 'bin' / 'python')


class ROMPEndToEndAuditTests(unittest.TestCase):
    def fixture(self, root, probabilistic, *, daily_rates=False, parallel=False, country='Ethiopia'):
        external = root / 'external'
        model_folder = 'AIFS_ENS_v2' if probabilistic else 'AIFS_v2'
        observations = (external / 'CHIRPS_regridded_025' / country if country == 'Ethiopia' else
                        external / 'CHIRPSv3_Country_0p25deg' / country)
        model = (external / ('aifs_ens' if probabilistic else 'aifs_single') if country == 'Ethiopia' else
                 external / 'Forecasts_Country_0p25deg' / model_folder / 'tp' / country)
        boundary = external / 'shapefile'
        for folder in (observations, model, boundary):
            folder.mkdir(parents=True)
        coordinates = ({'lat': [8., 9.], 'lon': [38., 39.]} if country == 'Ethiopia' else
                       {'lat': [-12., -11.], 'lon': [-77., -76.]})
        for year in (2021, 2022, 2023):
            times = pd.date_range(f'{year}-05-01', f'{year}-07-31')
            amounts = np.zeros((len(times), 2, 2))
            amounts[times >= f'{year}-05-20'] = 10.
            rainfall = xr.DataArray(amounts, dims=('time', 'lat', 'lon'),
                                    coords={'time': times, **coordinates}, attrs={'units': 'mm/day'})
            observation_file = f'{year}.nc' if country == 'Ethiopia' else f'CHIRPSv3_{country}_{year}_box025.nc'
            xr.Dataset({'RAINFALL': rainfall}).to_netcdf(observations / observation_file)
        initialization = pd.to_datetime(['2023-05-15', '2023-05-18'])
        steps = np.arange(1, 37)
        if probabilistic:
            amounts = np.zeros((2, 3, 36, 2, 2))
            for index, lead in enumerate((5, 2)):
                for member, shift in enumerate((-1, 0, 1)):
                    amounts[index, member, lead + shift - 1:] = .01
            rainfall = xr.DataArray(amounts, dims=('init_time', 'member', 'step', 'lat', 'lon'),
                                    coords={'init_time': initialization, 'member': [0, 1, 2],
                                            'step': steps, **coordinates}, attrs={'units': 'm'})
        else:
            amounts = np.zeros((2, 36, 2, 2))
            for index, lead in enumerate((5, 2)):
                amounts[index, lead - 1:, 0, 0] = .01  # Perfect.
                amounts[index, lead:, 0, 1] = .01  # One day late, within tolerance.
                # [1,0] never forecasts onset: one miss per initialization.
                amounts[index, :, 1, 1] = .01  # Four-day/one-day early cases.
            rainfall = xr.DataArray(amounts, dims=('init_time', 'step', 'lat', 'lon'),
                                    coords={'init_time': initialization, 'step': steps, **coordinates},
                                    attrs={'units': 'm'})
        if daily_rates:
            rainfall = rainfall * 1000
            rainfall.attrs['units'] = 'mm/day'
        model_file = '2023.nc' if country == 'Ethiopia' else f'{model_folder}_tp_{country}_2023_box025.nc'
        xr.Dataset({'tp': rainfall}).to_netcdf(model / model_file)
        mask = external / 'jjas_100mm_rainfall_mask_0p25.nc'
        xr.Dataset({'mask': xr.DataArray(np.ones((2, 2)), dims=('lat', 'lon'),
                                         coords=coordinates)}).to_netcdf(mask)
        box = ([[37, 7], [40, 7], [40, 10], [37, 10], [37, 7]] if country == 'Ethiopia' else
               [[-78, -13], [-75, -13], [-75, -10], [-78, -10], [-78, -13]])
        boundary.joinpath('ETH_Zones.geojson' if country == 'Ethiopia' else 'Peru.geojson').write_text(json.dumps({
            'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {},
            'geometry': {'type': 'Polygon', 'coordinates': [
                box]}}]}))
        shared = {'wet_day_threshold': 20., 'minimum_wet_day_rainfall': 1.,
                  'wet_spell_length': 3, 'dry_spell_limit': 3,
                  'dry_spell_search_extension': 5, 'area_mask_file': str(mask) if country == 'Ethiopia' else '',
                  'onset_threshold_file': '', 'baseline_forecast': 'climatology'}
        schedule = {'eval_start': '2023-05-15', 'eval_end': '2023-05-18',
                    'baseline_start_year': 2021, 'baseline_end_year': 2022,
                    'initialization_days': list(range(7)), 'init_type': 'daily',
                    'date_filter_year': 2023, 'run_years_concurrently': parallel,
                    'ensemble_forecast': probabilistic, 'max_forecast_day': 15,
                    'verification_window_list': [[1, 15]], 'tolerance_days_list': [3]}
        key = 'AIFS_ENS' if probabilistic else 'AIFS'
        spec = root / 'spec.json'
        spec.write_text(json.dumps({'forecast_window': '1-15', 'shared': shared,
                                    'models': [{'exec_key': key, **schedule}]}))
        return key, spec, {**shared, **schedule, 'forecast_window': '1-15', 'benchmark_coverage': country}

    def run_interfaces(self, root, probabilistic, *, daily_rates=False, parallel=False, country='Ethiopia'):
        key, spec, settings = self.fixture(root, probabilistic, daily_rates=daily_rates, parallel=parallel, country=country)
        env = dict(os.environ)
        # A developer's shell overrides must not change this controlled experiment.
        for name in list(env):
            if name.startswith('ROMP_'):
                del env[name]
        env.update(DEMO_DATA_DIR=str(root), DEMO_DATA_SOURCE='local',
                   ROMP_REGION=country,
                   ROMP_PYTHON=PYTHON, MPLCONFIGDIR=str(ROOT / '.mplcache'), MPLBACKEND='Agg',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        cli_env = {**env, 'ROMP_OUT_DIR': str(root / 'cli'), 'ROMP_FIG_DIR': str(root / 'cli-figures')}
        if country != 'Ethiopia':
            env['ROMP_SHAPEFILE_URI'] = str(root / 'external/shapefile/Peru.geojson')
            cli_env.update(env)
            cli_env['ROMP_OUT_DIR'] = str(root / 'cli/countries' / country)
            cli_env['ROMP_FIG_DIR'] = str(root / 'cli-figures/countries' / country)
        cli = subprocess.run([PYTHON, str(ROOT / 'run_momp.py'), '--mode',
                              'prob' if probabilistic else 'det', '--model', key, '--spec', str(spec)],
                             cwd=ROOT, env=cli_env, capture_output=True, text=True, timeout=120)
        self.assertEqual(cli.returncode, 0, (cli.stdout + cli.stderr)[-12000:])
        # Exercise the same process bridge used by the dashboard, including its
        # environment construction and runtime/dependency probe, with actual workers.
        from ui.pages.short_medium_benchmarking import run_momp_benchmark
        ui_env = {**env, 'ROMP_OUT_DIR': str(root / 'dashboard'),
                  'ROMP_FIG_DIR': str(root / 'dashboard-figures')}
        with patch.dict(os.environ, ui_env, clear=True):
            dashboard = run_momp_benchmark('prob' if probabilistic else 'det', key, settings,
                                            progress=lambda line: None)
        self.assertTrue(dashboard['ok'], dashboard.get('stderr', '') + dashboard.get('stdout', '')[-12000:])
        for interface in ('cli', 'dashboard'):
            self.assertTrue(list((root / f'{interface}-figures').rglob('*.png')),
                            f'{interface} did not produce actual figures')
        if country != 'Ethiopia':
            return root / 'cli/countries' / country, root / 'dashboard/countries' / country
        return root / 'cli', root / 'dashboard'

    def assert_netcdf_parity(self, cli, dashboard):
        files = sorted(path.relative_to(cli) for path in cli.rglob('*.nc'))
        self.assertTrue(files)
        self.assertEqual(files, sorted(path.relative_to(dashboard) for path in dashboard.rglob('*.nc')))
        for relative in files:
            with self.subTest(file=relative), xr.open_dataset(cli / relative) as first, \
                    xr.open_dataset(dashboard / relative) as second:
                xr.testing.assert_allclose(first, second)

    def test_deterministic_cli_and_dashboard_match_analytic_spatial_scores(self):
        with tempfile.TemporaryDirectory(prefix='romp-e2e-det-') as folder:
            cli, dashboard = self.run_interfaces(Path(folder), probabilistic=False)
            self.assert_netcdf_parity(cli, dashboard)
            self.assertTrue((Path(folder) / 'cli-figures/aifs_single/panel_portrait_mae_far_mr_aifs_single_15day.png').is_file())
            with xr.open_dataset(cli / 'aifs_single/spatial_metrics_aifs_single_1-15.nc') as result:
                np.testing.assert_allclose(result.false_alarm_rate, [[0., 0.], [np.nan, .5]], equal_nan=True)
                np.testing.assert_allclose(result.miss_rate, [[0., 0.], [1., 0.]])
                np.testing.assert_allclose(result.mean_mae, [[0., 1.], [np.nan, 2.5]], equal_nan=True)

    def test_daily_rates_run_in_parallel_through_cli_and_dashboard(self):
        for probabilistic in (False, True):
            with self.subTest(probabilistic=probabilistic), tempfile.TemporaryDirectory(prefix='romp-e2e-rates-') as folder:
                cli, dashboard = self.run_interfaces(Path(folder), probabilistic=probabilistic,
                                                      daily_rates=True, parallel=True)
                self.assert_netcdf_parity(cli, dashboard)
                model = 'aifs_ens' if probabilistic else 'aifs_single'
                with xr.open_dataset(cli / f'{model}/spatial_metrics_{model}_1-15.nc') as result:
                    if probabilistic:
                        np.testing.assert_allclose(result.mean_mae, 0.)
                    else:
                        np.testing.assert_allclose(result.mean_mae, [[0., 1.], [np.nan, 2.5]], equal_nan=True)

    def test_peru_country_boxes_run_without_ethiopian_masks(self):
        for probabilistic in (False, True):
            with self.subTest(probabilistic=probabilistic), tempfile.TemporaryDirectory(prefix='romp-e2e-peru-') as folder:
                cli, dashboard = self.run_interfaces(Path(folder), probabilistic=probabilistic, country='Peru')
                self.assert_netcdf_parity(cli, dashboard)
                model = 'aifs_ens' if probabilistic else 'aifs_single'
                with xr.open_dataset(dashboard / f'{model}/spatial_metrics_{model}_1-15.nc') as result:
                    self.assertEqual(result.attrs['region'], 'Peru')
                    self.assertTrue((result.lat < 0).all())
                    self.assertTrue((result.lon < 0).all())
                    self.assertTrue(result.mean_mae.notnull().any())
                self.assertFalse((Path(folder) / 'dashboard' / model).exists())

    def test_probabilistic_cli_and_dashboard_match_with_disjoint_baseline_years(self):
        with tempfile.TemporaryDirectory(prefix='romp-e2e-prob-') as folder:
            cli, dashboard = self.run_interfaces(Path(folder), probabilistic=True)
            self.assert_netcdf_parity(cli, dashboard)
            for interface in ('cli', 'dashboard'):
                portrait = Path(folder) / f'{interface}-figures/aifs_ens/panel_portrait_mae_far_mr_aifs_ens_15day.png'
                self.assertTrue(portrait.is_file(), f'Missing ensemble onset portrait: {portrait}')
                self.assertGreater(portrait.stat().st_size, 1000)
            with xr.open_dataset(cli / 'aifs_ens/spatial_metrics_aifs_ens_1-15.nc') as errors:
                np.testing.assert_allclose(errors.mean_mae, 0.)
                np.testing.assert_allclose(errors.false_alarm_rate, 0.)
                np.testing.assert_allclose(errors.miss_rate, 0.)
            with xr.open_dataset(cli / 'aifs_ens/probabilistic_metrics_aifs_ens_1-15.nc') as grid:
                np.testing.assert_allclose(grid.fair_brier_score, 0., atol=1e-12)
                np.testing.assert_allclose(grid.fair_rps, 0., atol=1e-12)
                self.assertTrue(grid.fair_brier_skill_score.isnull().all())
                self.assertTrue(grid.fair_rps_skill_score.isnull().all())
            from ui.utils.metric_filters import load_probabilistic_distribution_stats
            distributions, errors = load_probabilistic_distribution_stats(
                dashboard, models=['AIFS Ensemble'], windows=['Days 1–15'])
            self.assertFalse(errors)
            self.assertFalse(distributions.empty)
            self.assertAlmostEqual(distributions.set_index('Metric').loc['Fair RPS', 'Mean'], 0., places=12)
            files = sorted(path.relative_to(cli) for path in cli.rglob('*.csv'))
            self.assertTrue(files)
            self.assertEqual(files, sorted(path.relative_to(dashboard) for path in dashboard.rglob('*.csv')))
            for relative in files:
                with self.subTest(file=relative):
                    pd.testing.assert_frame_equal(pd.read_csv(cli / relative), pd.read_csv(dashboard / relative))
            overall = pd.read_csv(cli / 'aifs_ens/overall_skill_scores_aifs_ens_1-15.csv').iloc[0]
            self.assertEqual(overall.Reference_Calendar_Year, 2023)
            self.assertEqual(overall.Requested_Reference_Years, '(2021, 2022)')
            from ui.utils.benchmark_report import (
                collect_benchmark_settings,
                collect_evaluated_models,
            )

            settings = collect_benchmark_settings(collect_evaluated_models(dashboard))
            self.assertTrue(settings)
            for recorded in settings:
                self.assertEqual(recorded['Baseline years'], '2021-2022')
                self.assertEqual(recorded['Reference calendar year'], 'Not applicable')
            self.assertAlmostEqual(overall.Fair_Brier_Score, 0., places=12)
            self.assertAlmostEqual(overall.Fair_RPS, 0., places=12)
            self.assertTrue(np.isnan(overall.Fair_Brier_Skill_Score))  # Perfect baseline: zero denominator.
            self.assertTrue(np.isnan(overall.Fair_RPS_Skill_Score))

    def test_parallel_aifs_ensemble_country_boxes_complete_through_both_interfaces(self):
        with tempfile.TemporaryDirectory(prefix='romp-e2e-parallel-country-') as folder:
            cli, dashboard = self.run_interfaces(Path(folder), probabilistic=True,
                                                 parallel=True, country='Peru')
            self.assert_netcdf_parity(cli, dashboard)
            for output in (cli, dashboard):
                with xr.open_dataset(output / 'aifs_ens/probabilistic_metrics_aifs_ens_1-15.nc') as grid:
                    self.assertEqual(grid.attrs['region'], 'Peru')
                    self.assertTrue(grid.fair_brier_score.notnull().any())
                    self.assertFalse(np.isinf(grid.to_array()).any())
