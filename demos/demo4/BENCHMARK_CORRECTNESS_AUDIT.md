# Benchmark report and correctness audit

The PDF's **Recorded Benchmark Settings** section now reads saved output metadata per model,
verification window and output file. It preserves different baseline periods and weekly reference
calendars instead of substituting current dashboard controls. Consecutive years are compressed
without hiding gaps. Daily and monthly schedules label the reference calendar year as not applicable.
The report also distinguishes configured evaluation years from actual evaluation years where recorded.

## Confirmed issues addressed

| Issue | Correction |
| --- | --- |
| Per-model baseline/calendar settings omitted from PDF | Read recorded NetCDF attributes and CSV metadata; show current controls separately. |
| CSV and probability grids omitted reference calendar metadata | Persist calendar year, schedule, initialization days and evaluation dates in new outputs. |
| Default-country report scanned other countries' nested results | Match map discovery's country-directory isolation. |
| Previously generated PDF survived a new run or country/settings changes | Clear cached reports after runs and bind download availability to settings and output directory. |
| Ground-truth overrides replaced a configured custom reference source | Preserve custom reference directory, file pattern and variable when selecting observations. |
| November–March seasons could silently produce all-missing observed onsets | Reject unsupported cross-year seasons in dashboard preflight, runtime validation and direct onset detection. |
| Models using different baselines/schedules shared an overwritten climatology file | Suppress known incompatible shared reference windows, warn in the dashboard/PDF, and disable optional shared-baseline skill/significance calculations for those results. |

The compatibility guard compares recorded evaluation/baseline years, seasonal dates, weekly calendars,
initialization schedules, tolerance, onset criteria, observation sources, threshold variables and masks.
An explicitly disabled mask is distinct from absent metadata. Model metrics remain available when
shared-reference comparisons are suppressed. Per-model probabilistic scores retain their paired
reference calculations.

## Verification

Regression tests exercise the real PDF generator, real NetCDF/CSV metadata, model/window-specific
year reporting, country isolation, custom source overrides and incompatible reference suppression.
Synthetic end-to-end runs exercise deterministic and ensemble forecasts through both CLI and
dashboard subprocesses, including analytic scores and exported report metadata.

## Remaining limits

- Historical files without saved metadata cannot reveal their original calendar or baseline years.
  Reports label these values **Not recorded**. Rerun the benchmark to produce complete metadata.
- Metadata compatibility does not establish identical processed initialization samples when older
  files omit actual sample coverage. Unknown settings cannot prove or disprove compatibility.
- The shared climatology file is still replaced by subsequent runs. The guard prevents known
  incompatible comparisons; retaining independent reference artifacts for every experiment remains
  a future storage change.
- Cross-year seasons need adjacent-year observation loading and explicit season-year semantics;
  they are rejected until that support exists.
- A dashboard custom-baseline name alone does not configure a new data source. A custom reference
  must already have its directory, variable and file pattern configured.
