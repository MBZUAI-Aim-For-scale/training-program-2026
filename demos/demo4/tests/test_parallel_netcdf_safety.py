"""Probabilistic reference calculations must isolate native NetCDF libraries."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE_ROOT = Path(__file__).resolve().parents[1]


class ParallelNetCDFSafetyTests(unittest.TestCase):
    def test_real_netcdf_climatology_matches_serial_without_native_crash(self):
        script = '''
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from momp.io import input as inputs
from momp.stats.climatology import compute_climatological_onset_dataset
from momp.stats.parallel import parallel_climatological_onset_dataset
from momp.stats import parallel

original_dataset = parallel.compute_climatological_onset_dataset

def isolated_dataset(**kwargs):
    assert os.getpid() != int(os.environ['ROMP_TEST_PARENT_PID']), 'Native NetCDF I/O must run in isolated processes'
    return original_dataset(**kwargs)

parallel.compute_climatological_onset_dataset = isolated_dataset

# Keep this fixture focused on file I/O and onset calculations, without external
# boundary or mask files. Spawned children apply the same spatial selection.
inputs.region_select = lambda dataset, **kwargs: dataset
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

if __name__ == '__main__':
    folder = Path(sys.argv[1])
    years = tuple(range(2000, 2006))
    for year in years:
        dates = pd.date_range(f'{year}-05-01', periods=50)
        rainfall = np.zeros((50, 2, 2))
        rainfall[year - 1998:] = 10.
        xr.Dataset({'rain': (('time', 'lat', 'lon'), rainfall)},
                   coords={'time': dates, 'lat': [8., 9.], 'lon': [38., 39.]}).assign_attrs(
                       fixture='parallel native I/O').to_netcdf(folder / f'{year}.nc', engine='netcdf4')
    settings = dict(obs_dir=str(folder), obs_file_pattern=('{}.nc',), obs_var='rain',
                    obs_unit_cvt=1., region='fixture', years_clim=years,
                    thresh_file=None, wet_threshold=20., wet_init=1., wet_spell=3,
                    dry_spell=0, dry_threshold=1., dry_extent=0,
                    start_date=(2000, 5, 1), end_date=(2005, 5, 31), fallback_date=None)
    expected = compute_climatological_onset_dataset(**settings)
    os.environ['ROMP_TEST_PARENT_PID'] = str(os.getpid())
    for iteration in range(3):
        actual = parallel_climatological_onset_dataset(**settings)
        xr.testing.assert_equal(actual, expected)
        assert actual.attrs['years_processed'] == list(years)
        assert actual.attrs['total_years'] == len(years)
    print('parallel NetCDF reference matches serial')
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'check_parallel_netcdf.py'
            path.write_text(script)
            environment = {**os.environ, 'DEMO_DATA_SOURCE': 'local',
                           'PYTHONFAULTHANDLER': '1', 'PYTHONWARNINGS': 'ignore',
                           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                           'MKL_NUM_THREADS': '1', 'MPLCONFIGDIR': str(Path(directory) / 'mpl'),
                           'PYTHONPATH': os.pathsep.join([
                               str(SOURCE_ROOT), str(SOURCE_ROOT / 'apps/benchmarking/ROMP')])}
            result = subprocess.run([sys.executable, '-X', 'faulthandler', str(path), directory],
                                    env=environment, cwd=SOURCE_ROOT,
                                    capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-15000:])
        self.assertIn('parallel NetCDF reference matches serial', result.stdout)
