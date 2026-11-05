# ROMP/MOMP audit continuation

The continuation reproduced three failures in the existing full regression suite:
two onset horizon boundary failures and a probabilistic CLI integration failure.

## Corrections

- `momp/stats/detect.py`: the confirmation interval includes the candidate day.
  A day-15 candidate with a 21-day confirmation interval needs data through day
  35. The previous horizon calculation shortened that interval to 20 days and
  missed a rejecting dry spell completed on the final available day.
- `momp/stats/score.py`: paired RPS now normalizes dates and coordinates before
  comparing category coverage, as sample matching already did. A forecast
  timestamp and a climatology date string previously caused a `KeyError`.
  Missing or nonfinite case keys fail explicitly before grouping.
- `momp/graphics/func_map.py`: day-of-year labels use the existing pandas calendar
  utilities, avoiding missing datetime imports and respecting leap years.
- `momp/graphics/panel_map_error.py`: the standalone plotting entry point uses
  the panel filename convention instead of an undefined model name.
- `momp/io/output.py`: nested-dictionary inspection directly reports depth and
  ordered unique keys at each level instead of calling nonexistent helpers.
- `momp/utils/printing.py`: missing NetCDF input raises `FileNotFoundError` with
  the directory searched, instead of a secondary `NameError`.

Regression coverage was added in `tests/test_romp_utility_audit.py` and
`tests/test_romp_scoring_audit.py`. The existing onset boundary tests reproduced
both horizon failures before the fix. The existing end-to-end tests run real
synthetic NetCDF inputs through the CLI and dashboard process bridge, compare
scientific outputs, and generate figures.

## Verification scope

Run the full suite with:

```sh
MPLCONFIGDIR=.mplcache .venv/bin/python -m unittest discover -s tests
```

ROMP and its changed integration modules pass Python compilation and isolated
Ruff fatal/undefined-name checks (`--isolated --select E9,F63,F7,F82`). The
repository Ruff configuration still uses unsupported underscore-form isort
options. A broader UI scan also reports existing unresolved annotation names in
`ui/components/spatial_map.py` and `ui/utils/geo.py`, and a missing Streamlit name
in `ui/utils/css.py`; those files are outside this continuation's ROMP changes.
Existing unrelated documentation whitespace warnings remain in the full diff.

Final result: 293 tests ran successfully, with one opt-in process-group test
skipped. Both deterministic and probabilistic CLI/dashboard end-to-end tests
passed. Whitespace checks pass for the ROMP changes and this report.

Production data, remote APIs, GPU inference, and the standalone multi-model map
script were not exercised. No dependencies were added.

## Daily observation rates

A production traceback exposed a rejected `mm/day` observation variable. The
shared converter now accepts daily millimetre rates when callers explicitly
supply `sample_duration_days`. The observation loader validates consecutive daily
timestamps and supplies one day, so daily values remain numerically unchanged
and become amounts labelled `mm`. Source metadata still overrides catalog
multipliers. Missing durations, unsupported rates, and irregular/subdaily
observation timestamps fail explicitly.

Tests cover rate spellings, duration validation, repeated conversion, and real
NetCDF observation loading. The CLI/dashboard end-to-end fixtures now declare
observations in `mm/day` and retain their analytic score assertions.

Verification: 295 full-suite tests ran successfully (one opt-in test skipped).
Both end-to-end tests also passed after changing their observation units to
`mm/day`. Targeted fatal/undefined-name lint, compilation, and whitespace checks
passed.

## gRPC process startup diagnostics

The app previously forced `GRPC_ENABLE_FORK_SUPPORT=1` and
`GRPC_POLL_STRATEGY=poll` in both configuration and ROMP's parallel module.
That registered native fork handlers even though ROMP pools explicitly use the
`spawn` context and each worker initializes its own clients. With active gRPC
threads, those handlers log `Other threads are currently calling into gRPC,
skipping fork() handlers`; the first native log can also emit the Abseil
pre-initialization stderr banner.

Configuration now defaults fork support to `0` before third-party imports and
preserves explicit environment overrides. The parallel module no longer
overwrites these settings or forces a polling backend. No stderr or general
error logging is filtered. Restart existing dashboard/CLI processes to apply
the setting before gRPC initialization; an explicit shell setting of
`GRPC_ENABLE_FORK_SUPPORT=1` still opts into the handlers.

`tests/test_grpc_process_startup.py` checks fresh-interpreter configuration and
override preservation, then launches a real spawned worker while the parent has
a gRPC channel open. It checks worker completion and absence of both reported
messages. The channel does not make an external request.

Verification: 297 full-suite tests ran successfully (one opt-in test skipped),
including the real gRPC/spawn check. Neither reported native diagnostic appeared
in the captured full-suite output. Targeted static checks and compilation passed.

References: [gRPC fork support](https://github.com/grpc/grpc/blob/master/doc/fork_support.md)
and [native fork handler implementation](https://github.com/grpc/grpc/blob/master/src/core/lib/iomgr/fork_posix.cc).
