"""Derive report model counts from stored results, independent of UI selection."""

import ast
import csv
from datetime import date
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


def collect_pdf_probability_tables(output_root):
    """Read saved pooled scores and per-cell distributions without optional engines."""
    import pandas as pd

    from ui.utils.metric_filters import (
        load_probabilistic_distribution_stats,
        load_probabilistic_metric_table,
    )

    table, _, score_errors = load_probabilistic_metric_table(output_root)
    scores = pd.DataFrame([
        {'Model': model, 'Lead Window': window, 'Metric': metric, 'Score': score}
        for (metric, window, model), score in table.items()
    ], columns=['Model', 'Lead Window', 'Metric', 'Score'])
    distributions, grid_errors = load_probabilistic_distribution_stats(output_root)
    return scores, distributions, score_errors + grid_errors


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
                        # Read one complete record: per-target reference membership
                        # can exceed 16 KB and must never be truncated mid-field.
                        metadata = next(csv.DictReader(source), {})
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
                'Country': _recorded_value(metadata, 'region', 'Region'),
                'Ground truth': _recorded_value(metadata, 'obs', 'observation_source', 'Ground_Truth'),
                'Wet-day minimum': _recorded_value(metadata, 'wet_init'),
                'Wet spell total': _recorded_value(metadata, 'wet_threshold'),
                'Wet spell length': _recorded_value(metadata, 'wet_spell'),
                'Threshold source': ('Scalar' if 'thresh_file' in metadata
                                     and str(metadata['thresh_file']).strip().lower() in {'', 'none'}
                                     else _recorded_value(metadata, 'thresh_file')),
                'Dry threshold': _recorded_value(metadata, 'dry_threshold'),
                'Dry spell length': _recorded_value(metadata, 'dry_spell'),
                'Dry confirmation length': _recorded_value(metadata, 'dry_extent'),
                'Reference policy': _recorded_value(metadata, 'reference_policy', 'Reference_Policy'),
                'Reference membership': _recorded_value(metadata, 'reference_membership', 'Reference_Membership'),
                'Reference years by verification year': _recorded_value(
                    metadata, 'reference_years_by_verification_year', 'Reference_Years_By_Verification_Year'),
                'Output': path.name,
            })
    return records


def summarize_benchmark_settings(records):
    """Condense saved settings without merging conflicting result sets."""
    unknown = {'', 'none', 'nan', 'not recorded', 'not_recorded', 'not applicable', 'unknown'}

    def text(value):
        value = str(value).strip() if value is not None else ''
        return 'Unknown' if value.lower() in unknown else value

    def literal(value):
        try:
            return ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return None

    def years(value):
        value = text(value)
        if value == 'Unknown':
            return value
        return re.sub(r'\s*[-–]\s*', '-', _format_years(value))

    def month_day(value):
        parts = literal(value)
        if not isinstance(parts, (tuple, list)) or len(parts) not in {2, 3}:
            try:
                parsed = date.fromisoformat(str(value))
            except ValueError:
                return 'Unknown'
            month, day = parsed.month, parsed.day
        else:
            month, day = parts[-2:]
        try:
            date(2000, month, day)
        except (TypeError, ValueError):
            return 'Unknown'
        months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
        return f'{day:02d} {months[month - 1]}'

    def initialization(record):
        mode = text(record.get('Initialization schedule')).lower()
        if mode == 'daily':
            return 'Daily'
        days = literal(record.get('Initialization days'))
        if type(days) is int:
            days = (days,)
        valid = isinstance(days, (tuple, list)) and days and all(type(day) is int for day in days)
        if mode == 'weekly':
            weekdays = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
            selected = (', '.join(weekdays[day] for day in sorted(set(days)))
                        if valid and all(0 <= day <= 6 for day in days) else 'weekdays unknown')
            calendar = text(record.get('Reference calendar year'))
            return f'Weekly: {selected}\n{calendar} reference calendar'
        if mode == 'monthly':
            selected = (', '.join(map(str, sorted(set(days))))
                        if valid and all(1 <= day <= 31 for day in days) else 'unknown')
            return f'Monthly: days {selected}'
        calendar = text(record.get('Reference calendar year'))
        return f'Schedule unknown\nCalendar: {calendar}' if calendar != 'Unknown' else 'Unknown'

    def reference(record):
        policy = _normalized(text(record.get('Reference policy')))
        membership = set(str(record.get('Reference membership', '')).split(';'))
        if policy == 'unknown':
            if 'includes_verification_year' in membership:
                policy = 'in_sample'
            elif 'leave_one_year_out' in membership:
                policy = 'leave_one_year_out'
            elif membership == {'held_out_verification_year'}:
                policy = 'held_out_verification_year'
        labels = {'leave_one_year_out': 'Leave-one-year-out',
                  'held_out_verification_year': 'Independent baseline',
                  'in_sample': 'In-sample baseline'}
        label = labels.get(policy, 'Unknown')
        baseline = text(record.get('Baseline forecast'))
        if label == 'Unknown' and baseline != 'Unknown':
            label = ('Climatology (method unknown)' if baseline.lower() == 'climatology'
                     else baseline.replace('_', ' '))
        mapping = literal(record.get('Reference years by verification year'))
        counts = set()
        leaked = False
        if isinstance(mapping, dict):
            for target, values in mapping.items():
                members = literal(values)
                if type(members) is int:
                    members = (members,)
                if not isinstance(members, (tuple, list)) or not all(type(year) is int for year in members):
                    counts.clear()
                    break
                try:
                    target = int(target)
                except (TypeError, ValueError):
                    counts.clear()
                    break
                counts.add(len(set(members)))
                leaked = leaked or target in members
        if leaked:
            label = 'In-sample baseline'
        return label, counts

    criteria = ('Country', 'Ground truth', 'Wet-day minimum', 'Wet spell total', 'Wet spell length', 'Threshold source',
                'Dry threshold', 'Dry spell length', 'Dry confirmation length')
    entries = []
    for record in records:
        raw_model = text(record.get('Model'))
        model = _MODEL_NAMES.get(_normalized(raw_model), (None, raw_model))[1]
        output = Path(str(record.get('Output', ''))).name.lower()
        result_type = ('Probability scores' if output.startswith('overall_skill_scores_')
                       else 'Onset errors' if output.startswith('spatial_metrics_') else 'Saved results')
        configured = years(record.get('Evaluation years'))
        actual = years(record.get('Actual evaluation years'))
        period = actual if actual != 'Unknown' else configured
        if actual != 'Unknown' and configured not in {'Unknown', actual}:
            period += f' (configured {configured})'
        first, last = month_day(record.get('Evaluation start')), month_day(record.get('Evaluation end'))
        if first != 'Unknown' or last != 'Unknown':
            period += f'\n{first}-{last} annually'
        policy, counts = reference(record)
        row = {'Model': model, 'Lead time': text(record.get('Window')).replace('–', '-'),
               'Evaluation period': period, 'Baseline years': years(record.get('Baseline years')),
               'Initialization': initialization(record), 'Reference': policy}
        known_criteria = {}
        for field in criteria:
            value = text(record.get(field))
            if value != 'Unknown':
                parsed = literal(value)
                known_criteria[field] = parsed if isinstance(parsed, (int, float)) else value
        known = dict(known_criteria)
        values = {'Evaluation years': configured, 'Actual evaluation years': actual,
                  'Evaluation start': first, 'Evaluation end': last,
                  'Baseline years': row['Baseline years'],
                  'Initialization schedule': text(record.get('Initialization schedule')).lower(),
                  'Baseline forecast': text(record.get('Baseline forecast')).lower()}
        mode = values['Initialization schedule']
        days = literal(record.get('Initialization days'))
        if type(days) is int:
            days = (days,)
        if mode in {'weekly', 'monthly'} and isinstance(days, (tuple, list)):
            display_days = initialization(record).split('\n')[0].partition(':')[2].strip()
            if display_days and 'unknown' not in display_days:
                values['Initialization days'] = display_days
        if mode not in {'daily', 'monthly'}:
            values['Reference calendar year'] = text(record.get('Reference calendar year'))
        if 'unknown' not in policy.lower():
            values['Reference policy'] = policy
        known.update({field: value for field, value in values.items() if str(value).lower() != 'unknown'})
        entries.append({'row': row, 'results': {result_type}, 'counts': counts,
                        'criteria': known_criteria, 'known': known})

    def merge(target, entry):
        target['results'].update(entry['results'])
        target['counts'].update(entry['counts'])
        target['criteria'].update(entry['criteria'])
        target['known'].update(entry['known'])

    buckets = {}
    for entry in entries:
        buckets.setdefault(tuple(entry['row'].items()), []).append(entry)
    groups = []
    for bucket in buckets.values():
        variants = []
        for entry in bucket:
            same = next((item for item in variants if item['criteria'] == entry['criteria']), None)
            if same is None:
                variants.append(entry)
            else:
                merge(same, entry)
        conflicting = any(len({item['criteria'][field] for item in variants if field in item['criteria']}) > 1
                          for field in criteria)
        if not conflicting:
            for entry in variants[1:]:
                merge(variants[0], entry)
            groups.append(variants[0])
            continue
        # Establish all detailed variants before assigning sparse metadata. This
        # prevents input ordering from attaching an unknown record to one of two
        # incompatible experiments merely because it was encountered first.
        detailed = [entry for entry in variants if not any(
            set(entry['criteria'].items()) < set(other['criteria'].items()) for other in variants)]
        groups.extend(detailed)
        for entry in variants:
            if any(entry is group for group in detailed):
                continue
            matches = [group for group in detailed if all(
                group['criteria'][field] == value for field, value in entry['criteria'].items()
                if field in group['criteria'])]
            if len(matches) == 1:
                merge(matches[0], entry)
            else:
                groups.append(entry)

    issues = []
    scopes = {}
    for group in groups:
        scopes.setdefault((group['row']['Model'], group['row']['Lead time']), []).append(group)
    for (model, window), variants in scopes.items():
        if len(variants) > 1:
            details = '; '.join(f"{' and '.join(sorted(group['results']))}: "
                                f"{group['row']['Evaluation period'].replace(chr(10), '; ')}"
                                for group in variants)
            fields = dict.fromkeys(field for group in variants for field in group['known'])
            changed = [field for field in fields
                       if len({str(group['known'][field]) for group in variants if field in group['known']}) > 1]
            changes = []
            for field in changed:
                if field in {'Evaluation years', 'Actual evaluation years', 'Evaluation start', 'Evaluation end'}:
                    continue
                values = '; '.join(
                    f"{' and '.join(sorted(group['results']))}: "
                    f"{group['known'].get(field, 'Unknown')}"
                    for group in variants)
                changes.append(f'{field} ({values})')
            suffix = f" Differing settings: {', '.join(changes)}." if changes else ''
            if changed:
                issues.append(f'{model} ({window}) has conflicting saved result settings. {details}.{suffix} '
                              'These result sets are reported separately; rerun matching settings before combining them.')
            else:
                issues.append(f'{model} ({window}) has incomplete saved settings; a common experiment cannot be confirmed. '
                              f'{details}. These result sets are reported separately until matching settings are recorded.')
        if any(group['row']['Reference'] == 'In-sample baseline' for group in variants):
            issues.append(f'{model} ({window}) includes a verification year in its reference. '
                          'Rerun with an independent baseline before interpreting reference skill.')
    rows = []
    for group in sorted(groups, key=lambda item: (item['row']['Model'], item['row']['Lead time'],
                                                 item['row']['Evaluation period'])):
        row = group['row']
        counts = sorted(group['counts'])
        reference_text = row['Reference']
        if counts:
            size = str(counts[0]) if counts[0] == counts[-1] else f'{counts[0]}-{counts[-1]}'
            reference_text += f'\n{size} eligible years'
        rows.append({'Model': row['Model'], 'Results': '\n'.join(sorted(group['results'])),
                     'Lead time': row['Lead time'], 'Evaluation period': row['Evaluation period'],
                     'Baseline years': row['Baseline years'], 'Initialization': row['Initialization'],
                     'Reference': reference_text})
    return rows, issues


def _reference_differences(metadata, reference_metadata):
    """Return recorded differences between one model and its reference."""
    fields = ('years', 'years_clim', 'start_date', 'end_date', 'init_type', 'init_days',
              'date_filter_year', 'tolerance_days', 'wet_init', 'wet_threshold', 'wet_spell',
              'dry_threshold', 'dry_spell', 'dry_extent', 'obs_dir', 'obs_var',
              'obs_file_pattern', 'obs_unit_cvt', 'thresh_file', 'thresh_var', 'mok', 'nc_mask', 'reference_policy')

    missing = object()

    def value(metadata, key):
        aliases = {'years': 'evaluation_years', 'years_clim': 'reference_years'}
        if key not in metadata:
            key = aliases.get(key, key)
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

    def label(value, field):
        if value is missing:
            return 'Not recorded'
        if field in {'years', 'years_clim'}:
            return _format_years(str(value))
        if field in {'start_date', 'end_date'} and isinstance(value, tuple) and len(value) == 3:
            return '-'.join(str(part).zfill(2) for part in value)
        return str(value)

    differences = []
    for field in fields:
        modes = {str(values.get('init_type', '')).lower() for values in (metadata, reference_metadata)}
        if field == 'date_filter_year' and modes <= {'daily', 'monthly'}:
            continue
        if field == 'init_days' and modes == {'daily'}:
            continue
        left, right = value(metadata, field), value(reference_metadata, field)
        unknown_policy = field == 'reference_policy' and (left is missing) != (right is missing)
        if unknown_policy or (left is not missing and right is not missing and left != right):
            name = {'years': 'Evaluation years', 'years_clim': 'Baseline years',
                    'start_date': 'Evaluation start', 'end_date': 'Evaluation end'}.get(
                        field, field.replace('_', ' ').title())
            differences.append(f'{name}: model={label(left, field)}; climatology={label(right, field)}')
    return differences


def reference_comparison_allowed(left, left_window, right, right_window):
    """Keep raw model comparisons available while guarding model/reference pairs."""
    left_is_reference = _normalized(left['model_label']) in _BASELINES
    right_is_reference = _normalized(right['model_label']) in _BASELINES
    if left_is_reference == right_is_reference:
        return True
    forecast, window, reference = ((right, right_window, left) if left_is_reference
                                    else (left, left_window, right))
    blocked = {str(value).replace('–', '-') for value in forecast.get('reference_incompatible_windows', ())}
    return (str(window).replace('–', '-') not in blocked
            and not _reference_differences(forecast.get('metadata', {}), reference.get('metadata', {})))


def compatible_reference_outputs(outputs):
    """Keep reference windows useful to matching models; flag each conflicting model."""
    conflicts = {}
    reference_windows = {}
    for reference_index, reference in enumerate(outputs):
        if _normalized(reference['model_label']) not in _BASELINES:
            continue
        keep = set(reference['windows'])
        for window in reference['windows']:
            candidates = [(index, forecast) for index, forecast in enumerate(outputs)
                          if _normalized(forecast['model_label']) not in _BASELINES
                          and window in forecast['windows']]
            matches = False
            for index, forecast in candidates:
                differences = _reference_differences(forecast.get('metadata', {}), reference.get('metadata', {}))
                if differences:
                    conflicts.setdefault(index, {})[window] = differences
                else:
                    matches = True
            if candidates and not matches:
                keep.discard(window)
        reference_windows[reference_index] = keep
    if not conflicts:
        return outputs
    filtered = []
    for index, output in enumerate(outputs):
        item = dict(output)
        if index in reference_windows:
            item['windows'] = [window for window in output['windows'] if window in reference_windows[index]]
            if not item['windows']:
                continue
        if index in conflicts:
            item['reference_incompatible_windows'] = sorted(conflicts[index])
            details = ' '.join(f"{window}: {', '.join(differences)}."
                               for window, differences in sorted(conflicts[index].items()))
            item['reference_warning'] = (
                f"Shared climatology comparisons omitted for {output['model_label']} because of different benchmark settings. "
                f"{details} Rerun this model with matching settings to regenerate its comparison.")
        filtered.append(item)
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
