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

Compatibility is scoped to each model and verification window. A stale model no longer removes
a shared climatology window that matches another model. Warnings name the affected model, exact
fields and both recorded values. Raw maps/metrics remain visible; incompatible shared-reference
table colors, distribution fallback and Difference/Swipe pairs are disabled only for that model.
Matching models and saved paired probability references remain usable. Optional aggregate
skill/significance engines still use the conservative all-results guard when their inputs cannot
be scoped to matching models. Reports retain shared climatology figures when a compatible
model/reference pair remains available.

All Metrics lead-time choices now come from saved scores rather than a forced Days 1–15/Days 16–30
pair. Tables omit model/window columns with no finite values and do not synthesize missing result
rows for configured models. Missing or undefined scores within a populated column remain visible
as `N/A`; their presence does not imply that an uncomputed lead window should be displayed.

## Verification

Regression tests exercise the real PDF generator, real NetCDF/CSV metadata, model/window-specific
year reporting, country isolation, custom source overrides and incompatible reference suppression.
Synthetic end-to-end runs exercise deterministic and ensemble forecasts through both CLI and
dashboard subprocesses, including analytic scores and exported report metadata.

PDF tables now wrap complete headers and values at readable sizes, allocate widths from content,
repeat Model/Metric/Lead identifiers across wide continuations, and repeat headers after page breaks.
Oversized multiline rows can continue across pages. Named or hierarchical row indexes become visible
columns, duplicate column labels are preserved with unique names, and hierarchical column headers
are flattened. Small nonzero values use scientific notation rather than rounding to zero; comparison
symbols are transliterated without losing their meaning. The old clipping helper was removed.
Saved probabilistic CSV scores and grid distributions are included without the optional verification
engine. Read errors are recorded in the report. Cached PDFs using the old table layout must be regenerated.

PDF regressions cover duplicate/hierarchical labels, long text, wide and tall tables, repeated identities,
numeric precision, and full text extraction. Rendered sample pages are inspected and checked against
page margins; the preview artifact uses explicitly labelled synthetic data.

The main PDF now summarizes saved setup in a compact table rather than printing raw per-file
metadata. Equivalent CSV/spatial settings are combined; conflicting periods or onset criteria remain
separate and are flagged. Dates, weekdays and reference policies use readable labels. Detailed
reference-year mappings remain in the saved scientific outputs and are reduced to eligible-year
counts in the report. Current dashboard controls and source-file listings are no longer repeated
through the scientific narrative. Recorded onset criteria are shown separately from current controls.

## Leave-one-year-out verification

Climatology verification now uses **leave-one-year-out** membership in probabilistic and spatial
metrics. For example, verifying 2019 with requested baseline 2007–2024 excludes 2019 and retains
the other 17 eligible years. Observed 2019 onset remains the truth. If the target year is outside
the baseline, membership is unchanged. Baseline onset dates are computed once and reused to
derive target-specific held-out reference means.

New CSV/probability-grid exports record `Reference_Policy`, membership and eligible reference years
per verification year. Spatial metrics record the exclusion policy, and climatology onset exports
retain a `verification_year` dimension with per-target membership. A displayed average of those
fields is labelled as descriptive; it is not used in place of the per-target forecasts for scoring.
Finite-onset conditioning and fair-score sample exclusions still apply after holding out the target.

An overlapping probabilistic baseline needs at least three eligible years so at least two remain
after exclusion; a deterministic mean needs at least one remaining year. Empty references raise
an error rather than falling back to the excluded year. Existing saved results must be recomputed;
regenerating only the PDF cannot remove leakage from old scores. Historical cross-validation may
still use years later than the target; a past-only operational comparison needs a baseline entirely
before the evaluation period.

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
