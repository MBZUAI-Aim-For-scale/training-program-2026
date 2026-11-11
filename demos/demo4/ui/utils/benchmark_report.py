"""Derive report model counts from stored results, independent of UI selection."""

import ast
import csv
from pathlib import Path
import re

_OUTPUT_NAME = re.compile(
    r'^(?P<kind>spatial_metrics|overall_skill_scores)_(?P<model>.+)_'
    r'(?P<first>\d+)[_-](?P<last>\d+)\.(?P<extension>nc|csv)$', re.IGNORECASE)
_MODEL_NAMES = {
    'aifs': ('aifs_single', 'AIFS Single'),
    'aifs_single': ('aifs_single', 'AIFS Single'),
    'aifs_ens': ('aifs_ens', 'AIFS Ensemble'),
    'aifs_ensemble': ('aifs_ens', 'AIFS Ensemble'),
    'fuxi': ('fuxi', 'FuXi'),
    'graphcast': ('graphcast', 'GraphCast'),
    'gencast': ('gencast', 'GenCast'),
}
_BASELINES = {'climatology', 'traditional_climatology', 'traditional', 'baseline', 'reference', 'ref'}
_METRIC_COLUMNS = {'fair_brier_score', 'fair_brier_skill_score', 'fair_rps',
                   'fair_rps_skill_score', 'auc'}


def _recorded_value(metadata, *keys):
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip().lower() not in {
            '', 'none', 'nan', 'not_recorded', 'not recorded', '()', '[]',
        }:
            return str(value)
    return 'Not recorded'


def _format_years(value):
    """Compress consecutive years without concealing gaps in the recorded set."""
    if value == 'Not recorded':
        return value
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, int):
            parsed = (parsed,)
        years = sorted({int(year) for year in parsed})
    except (ValueError, SyntaxError, TypeError):
        return value
    spans = []
    for year in years:
        if spans and year == spans[-1][1] + 1:
            spans[-1][1] = year
        else:
            spans.append([year, year])
    return ', '.join(str(first) if first == last else f'{first}-{last}'
                     for first, last in spans) or 'Not recorded'


def collect_benchmark_settings(model_rows):
    """Read output metadata only; dashboard selections are never provenance."""
    records = []
    for model in model_rows:
        for filename in model.get('_output_paths', ()):
            path = Path(filename)
            match = _OUTPUT_NAME.match(path.name)
            if match is None:
                continue
            metadata = {}
            try:
                if path.suffix.lower() == '.csv':
                    with path.open(encoding='utf-8-sig', newline='') as source:
                        metadata = next(csv.DictReader(source.read(16384).splitlines()), {})
                else:
                    import xarray as xr

                    with xr.open_dataset(path, decode_cf=False, create_default_indexes=False) as dataset:
                        metadata = dict(dataset.attrs)
            except (OSError, ValueError, UnicodeError, csv.Error):
                pass
            mode = _recorded_value(metadata, 'init_type', 'Init_Type')
            calendar = _recorded_value(metadata, 'date_filter_year', 'Reference_Calendar_Year')
            if mode.lower() in {'daily', 'monthly'}:
                calendar = 'Not applicable'
            records.append({
                'Model': model.get('Model', match['model']),
                'Window': f"Days {match['first']}-{match['last']}",
                'Evaluation years': _format_years(_recorded_value(
                    metadata, 'years', 'evaluation_years', 'Evaluation_Years')),
                'Actual evaluation years': _format_years(_recorded_value(
                    metadata, 'Actual_Evaluation_Years', 'actual_evaluation_years')),
                'Baseline years': _format_years(_recorded_value(
                    metadata, 'years_clim', 'reference_years', 'Requested_Reference_Years', 'Reference_Years')),
                'Reference calendar year': calendar,
                'Initialization schedule': mode,
                'Initialization days': _recorded_value(metadata, 'init_days', 'Init_Days'),
                'Evaluation start': _recorded_value(metadata, 'start_date', 'Evaluation_Start'),
                'Evaluation end': _recorded_value(metadata, 'end_date', 'Evaluation_End'),
                'Baseline forecast': _recorded_value(metadata, 'ref_model', 'Baseline_Forecast'),
                'Output': path.name,
            })
    return records


def compatible_reference_outputs(outputs):
    """Suppress shared climatology windows whose recorded experiment differs."""
    fields = ('years', 'years_clim', 'start_date', 'end_date', 'init_type', 'init_days',
              'date_filter_year', 'tolerance_days', 'wet_init', 'wet_threshold', 'wet_spell',
              'dry_threshold', 'dry_spell', 'dry_extent', 'obs_dir', 'obs_var',
              'obs_file_pattern', 'obs_unit_cvt', 'thresh_file', 'thresh_var', 'mok', 'nc_mask')

    missing = object()

    def value(metadata, key):
        if key not in metadata:
            return missing
        raw = metadata.get(key)
        if raw is None or str(raw).strip().lower() == 'none':
            return None
        if str(raw).strip().lower() in {'', 'not_recorded'}:
            return missing
        try:
            parsed = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            return str(raw).strip()
        return tuple(parsed) if isinstance(parsed, (list, tuple)) else parsed

    incompatible = set()
    for reference in outputs:
        if _normalized(reference['model_label']) not in _BASELINES:
            continue
        reference_metadata = reference.get('metadata', {})
        for forecast in outputs:
            if _normalized(forecast['model_label']) in _BASELINES:
                continue
            metadata = forecast.get('metadata', {})
            common_windows = set(reference['windows']) & set(forecast['windows'])
            if not common_windows:
                continue
            for field in fields:
                if field == 'date_filter_year' and (
                    str(metadata.get('init_type', '')).lower() in {'daily', 'monthly'}
                    and str(reference_metadata.get('init_type', '')).lower() in {'daily', 'monthly'}
                ):
                    continue
                left, right = value(metadata, field), value(reference_metadata, field)
                if left is not missing and right is not missing and left != right:
                    incompatible.update(common_windows)
                    break
    if not incompatible:
        return outputs
    warning = ('Shared climatology comparisons were omitted for '
               + ', '.join(sorted(incompatible))
               + ' because the saved models and reference use different benchmark settings. '
               'Use matching evaluation years, baseline years, schedules and onset criteria for shared comparisons.')
    filtered = []
    for output in outputs:
        windows = output['windows']
        if _normalized(output['model_label']) in _BASELINES:
            windows = [window for window in windows if window not in incompatible]
        if windows:
            filtered.append({**output, 'windows': windows, 'reference_warning': warning})
    return filtered


def _normalized(name):
    return re.sub(r'[^a-z0-9]+', '_', str(name).strip().lower()).strip('_')


def _has_result_content(path, kind):
    """Check bounded artifact headers without loading scientific fields."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if kind == 'spatial_metrics':
            with path.open('rb') as source:
                header = source.read(8)
            return header[:4] in (b'CDF\x01', b'CDF\x02', b'CDF\x05') or header == b'\x89HDF\r\n\x1a\n'
        with path.open(encoding='utf-8-sig', newline='') as source:
            # Overall score tables are small. Bound reads even if an unrelated
            # large CSV has been given a matching filename.
            rows = csv.reader(source.read(16384).splitlines())
            columns = {_normalized(column) for column in next(rows, [])}
            values = next(rows, [])
        return bool(any(value.strip() for value in values)) and (
            bool(columns & _METRIC_COLUMNS) or {'metric', 'score'} <= columns)
    except (OSError, UnicodeError, csv.Error):
        return False


def collect_evaluated_models(output_dir, model_catalog=None, inventory_rows=(), reference_models=()):
    """Return one report row per model with persisted scientific result evidence.

    Inventory/catalog values supply identity and display labels only. Current
    selections and source-data availability never establish that a model was
    evaluated. The returned output paths and windows make that distinction
    available to the report without opening NetCDF datasets.
    """
    aliases = dict(_MODEL_NAMES)
    metadata = {}
    for key, model in (model_catalog or {}).items():
        name = _normalized(model.get('model_name') or key)
        canonical, default_label = aliases.get(name, (name, str(model.get('model_name') or key)))
        label = _MODEL_NAMES.get(name, (canonical, default_label))[1]
        for alias in (key, name):
            aliases[_normalized(alias)] = (canonical, label)
        metadata[canonical] = {
            'key': key,
            'label': label,
            'probabilistic': bool(model.get('probabilistic', False)),
        }
    for row in inventory_rows:
        name = _normalized(row.get('_model_name') or row.get('_exec_key') or row.get('Model', ''))
        if not name:
            continue
        canonical, default_label = aliases.get(name, (name, str(row.get('Model') or name)))
        label = row.get('Model') or default_label
        for alias in (name, row.get('_exec_key', ''), row.get('_row_key', ''), label):
            if alias:
                aliases[_normalized(alias)] = (canonical, label)
        metadata[canonical] = {
            'key': row.get('_exec_key', canonical), 'label': label,
            'probabilistic': bool(row.get('_probabilistic', row.get('Category') == 'Probabilistic')),
        }

    excluded = _BASELINES | {_normalized(name) for name in reference_models if name}
    excluded |= {aliases.get(name, (name, ''))[0] for name in tuple(excluded)}
    evaluated = {}
    for path in sorted(Path(output_dir).rglob('*')):
        if path.relative_to(output_dir).parts[0] == 'countries':
            continue
        match = _OUTPUT_NAME.match(path.name)
        if match is None:
            continue
        kind, extension = match['kind'].lower(), match['extension'].lower()
        if (kind, extension) not in {('spatial_metrics', 'nc'), ('overall_skill_scores', 'csv')}:
            continue
        first, last = int(match['first']), int(match['last'])
        if first < 1 or last < first:
            continue
        name = _normalized(match['model'])
        canonical, label = aliases.get(name, (name, name.replace('_', ' ').title()))
        if name in excluded or canonical in excluded or not _has_result_content(path, kind):
            continue
        model = metadata.get(canonical, {})
        entry = evaluated.setdefault(canonical, {
            'Model': model.get('label', label),
            'Category': 'Deterministic', 'Type': 'Stored benchmark results',
            'Coverage': 'Not recorded', 'Date Range': 'Not recorded',
            'Provenance': 'Existing benchmark output files',
            '_model_name': canonical, '_exec_key': model.get('key', canonical),
            '_probabilistic': False, '_output_paths': [], '_output_windows': [],
        })
        if kind == 'overall_skill_scores' or model.get('probabilistic'):
            entry['Category'] = 'Probabilistic'
            entry['_probabilistic'] = True
        entry['_output_paths'].append(str(path))
        window = f'Days {first}–{last}'
        if window not in entry['_output_windows']:
            entry['_output_windows'].append(window)

    return sorted(evaluated.values(), key=lambda row: row['Model'])
