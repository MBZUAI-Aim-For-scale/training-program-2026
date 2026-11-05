# Scientific comparison with the supplied ROMP package

Reference supplied by the project owner:
`/Users/mesfind/opt/Research/uchicago/ETAgroMetAIx/HCWF/ETAgroMetAIx/ROMP/momp`.
The dashboard continues to run its vendored package under
`apps/benchmarking/ROMP/momp`; the reference was read without modification.
`romp-reference-manifest.json` records the compared source fingerprints.
Matching a reference implementation establishes compatibility for tested cases;
it does not validate the physical interpretation of every input dataset.

## Confirmed discrepancies corrected

| Area | Discrepancy | Correction and evidence |
| --- | --- | --- |
| Confirmation horizon | Short forecasts silently shortened the configured dry-confirmation interval, while observations kept the original criterion. | Preserve the requested interval. A day-15 candidate with a 21-day interval needs data through day 35; insufficient coverage fails. Disabling dry checking requires only the wet-spell tail. Analytic horizon tests cover all cases. |
| Ensemble spatial metrics | Individual members were scored as independent spatial forecasts; `onset_percentage_threshold` had no effect. | Apply the reference consensus fraction and rounded mean onset day before spatial verification. One onset in three members at threshold 0.5 is one missed ensemble event; two qualifying members at days 2 and 6 produce one day-4 event. Both spatial paths are tested. |
| Timing contingency | An in-window mistimed event counted as both FP and FN, changing the reference definition and rate denominators. | Restore the reference exclusive FP outcome. Tests check one contingency outcome per initialization and corresponding spatial rates. |
| Coordinates | Forecast, observation, and threshold arrays could be paired by positional indices, including different physical cells. | Reorder identical named coordinates; reject different, duplicate, or nonfinite grids. No interpolation or implicit regridding is introduced. Tests cover reordered grids and incompatible threshold domains. |
| Rainfall variable | A lone unrelated variable could be selected as rainfall when the configured name was absent. | Require an explicitly configured name or recognized precipitation alias; do not treat temperature as rain. |
| Parameters | The independent dry-day threshold was not forwarded by dashboard/spec/environment bridges; spell widget defaults were hardcoded. | Preserve `dry_threshold` through the UI, spec, environment, notebook bridge, and runtime config. Widget defaults honor configured spell lengths. |
| Model specification | A nonempty model spec that omitted the requested model silently fell back to defaults. | Reject unmatched supplied model lists. Shared-only specs remain supported. |
| Ranked probability categories | RPS could change when the same configured bins were listed in another order. | Canonicalize chronological generation and runtime bins. Direct RPS rejects inconsistent category chronology/partition bounds. Permutation tests preserve analytic RPS. |

## Definitions preserved and tested

- Wet days use `rainfall >= wet_init`; the wet-spell total must be strictly
  greater than `wet_threshold`. The independent dry threshold is used for dry
  days. Lead day 1 maps to initialization date plus one day.
- Spatial ensemble onset uses the reference consensus fraction and Python's
  rounded mean among members that predict onset. Probability-bin scores continue
  to use individual usable members, including usable members with no onset.
- Ordinary and fair Brier scores, ordinary and fair RPS, and AUC are compared
  directly with the supplied reference on valid analytic inputs.
- Actual reference execution is also compared against four gridded ensemble
  fixtures, including below-threshold consensus and rounded timing.

Reference-execution tests use `ROMP_REFERENCE_MOMP` when provided; otherwise they
look for the supplied sibling checkout. These checks skip only when that checkout
is unavailable. Analytic expected-value tests remain portable without it.

## Deliberate departures retained

Some differences from the reference fix independently reproduced defects rather
than adapt its mathematics to the dashboard:

- Each forecast uses its own observed date instead of the first row's date.
  Verification is specific to the requested interval; earlier events are
  classified as non-events for that interval rather than omitted rows.
- Forecast/reference skill scores use matched cases, and RPS uses a complete,
  ordered, disjoint partition including the before/after-window categories.
  Zero reference-score denominators yield undefined skill, not infinity.
- Missing rainfall is excluded from usable members rather than interpreted as a
  valid no-onset forecast. Fair corrections require adequate effective members.
- Parallel climatology reduces individual yearly values rather than averaging
  chunk means, which biased uneven chunks and spatially missing years.
- The inclusive confirmation interval avoids the reference's extra horizon day;
  it retains the entire requested scientific interval.
- Explicit precipitation metadata determines scaling. Daily `mm/day` observations
  integrate over a validated one-day sample. Unsupported rates/subdaily inputs
  fail instead of receiving an arbitrary catalog multiplier.

These differences have analytic regressions; the dashboard is consequently not
claimed to be byte-for-byte or output-for-output identical to every reference
case, including cases where that reference has a reproduced defect.

Spatial NetCDF outputs now record the runtime wet/dry criteria, consensus
threshold, member selection, baseline years, initialization settings, input and
mask identifiers, and the contingency/aggregation definition. Score CSVs retain
requested years and separately record actual matched evaluation years where
available. These attributes describe the experiment; they do not establish that
an input's declared rainfall units or accumulation convention are physically
correct.

## Experiment choices and inherited limitations

The reference file and dashboard do not define the same default experiment:

| Setting | Reference config | Dashboard config |
| --- | --- | --- |
| Dry-confirmation interval | 0, disabled | 0, aligned with reference (previously 21) |
| Evaluation years | 2019–2024 | 2003–2023 |
| Climatology years | 1998–2024 | 2007–2024 |

The dry-confirmation default was subsequently aligned after directly checking
the reference against the reported 45-day forecast. Evaluation and climatology
years remain experiment-specific. To reproduce a reference experiment, explicitly
match the years, observation/forecast inputs,
mask, region, unit interpretation, schedules, thresholds, tolerances, member
selection, and verification windows. The full confirmation horizon must be
available for the chosen criteria.

Both implementations can include evaluation years in the climatology. That is
an in-sample reference, not held-out validation. Probabilistic climatology now
warns when a verification year is included and exports reference-membership
metadata; it does not silently substitute a leave-one-year-out experiment.
For held-out benchmarking, select disjoint climatology and evaluation years.

The reference also conditions climatological probabilities on finite detected
onsets, while forecast probabilities include usable no-onset members. This
inherited difference is recorded in exported probability-conditioning metadata.
Changing that convention needs an explicit scientific protocol, including how
to distinguish missing observations from a genuine season without onset.

Climatology uses raw day-of-year values, so leap-year calendar mapping follows
that convention. Cross-calendar-year onset seasons are not validated by this
audit. No production-data benchmark, GPU inference, or external API run was
used as evidence of scientific accuracy.

## Changed files and verification

Scientific changes are in `stats/detect.py`, `stats/benchmark.py`, `stats/bins.py`,
`stats/score.py`, `io/input.py`, `io/output.py`, and `utils/standard.py` inside the
vendored package. Parameter forwarding changes are in `run_momp.py`,
`notebooks/utils.py`, `ui/pages/short_medium_benchmarking.py`, and
`ui/utils/parameter_help.py`. New regression files are
`test_romp_reference_onset.py`, `test_romp_reference_scoring.py`,
`test_romp_reference_parameters.py`, and `test_romp_grid_integrity.py`; existing
input, probability, contingency, interface, and end-to-end tests were updated
where they encoded the corrected behavior.

The changes reuse the existing member-onset table and serializers; no new
dependencies or alternative scientific engine were introduced. Previously
generated outputs were not overwritten by the audit and must be recomputed
before interpreting them under these corrected definitions.

Final verification: 328 tests ran successfully, with one opt-in process-group
test skipped. Both supplied-reference comparison tests executed (not skipped),
and deterministic/probabilistic CLI-versus-dashboard synthetic NetCDF integration
tests passed. Python compilation, isolated Ruff fatal/undefined-name checks, and
whitespace checks passed for the changed code. The repository-wide Ruff config
still has pre-existing underscore-form isort settings; validation used
`ruff check --isolated --select E9,F63,F7,F82`.

## Forecasts ending at day 45

A reported AIFS ensemble/GenCast run requested verification through day 30 with
`wet_spell=3`, `dry_spell=7`, and `dry_extent=21`. Its required final rainfall
day is `30 + 21 - 1 = 50`; data ending at day 45 cannot confirm the entire last
candidate's interval. This is an incompatible experiment, not a reason to shorten
confirmation silently. Preserving 21-day confirmation supports candidates only
through day 25 on those data. The reference's disabled confirmation (`dry_extent=0`)
instead requires day 32 for a 30-day window because the wet spell still needs
three days.

The CLI now checks forecast lead metadata before invoking the scientific driver
or creating parallel onset workers. Lower-level probability-pair generation also
checks the horizon before loading observations. The dashboard explains available
versus required coverage and identifies the reference's disabled setting in the
dry-confirmation control. These checks do not choose different onset criteria
on the user's behalf.

Horizon follow-up verification: 334 tests ran successfully (one opt-in test
skipped). Five metadata-preflight regressions cover 45-versus-50-day rejection,
valid alternative criteria/windows, insufficient coverage in a later year, a
backend read guard, and rejection before driver invocation. Both real
CLI/dashboard synthetic integration tests passed. The observation-loading guard,
targeted lint, compilation, and whitespace checks also passed. Scientific
defaults and criteria were not changed by this follow-up.

## Aligning the default with the reference

The subsequent reference check confirmed `params/config.in:109` sets
`dry_extent=0`, while its verification window ends at day 30. The dashboard's
previous default of 21 was the reason that combination needed data beyond day
45. Configuration, package fallbacks, direct detector defaults, and UI helper
defaults now consistently use 0. The 3-day wet-spell criterion remains enabled,
so 30-day verification needs data through day 32.

Direct execution of both packages on a 45-day synthetic forecast confirms the
reference setting works and produces matching ensemble summaries. Both packages
reject an explicitly requested 21-day confirmation interval on that input.
Explicit spec/environment/widget settings remain authoritative: saved runs
containing 21 are not silently rewritten. Set their dry-confirmation value to 0
to run the reference experiment. Previously generated outputs are unchanged.

This default alignment changes `ai_weather/config.py`, vendored `params.py` and
`params/config.in`, `stats/detect.py`, and the dashboard/helper defaults.
Verification: 336 full-suite tests passed (one opt-in test skipped); all seven
reference-onset tests also passed after adding the reported 45-day case.
