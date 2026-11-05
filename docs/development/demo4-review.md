# Demo 4 reviewer follow-up

All reported findings were checked against the current checkout. Existing fixes
were retained and verified; remaining defects were corrected without changing
the scientific scoring definitions or regenerating production results.

| Review finding | Resolution | Evidence |
| --- | --- | --- |
| Host `.venv` instruction lacks Docker context | README, lesson, and both slide sources explicitly distinguish container `/opt/envs/app/bin/python` from an already configured host environment outside Docker. Docker participants do not need a host environment. | Confirmed the image's `VIRTUAL_ENV` and `PATH` in `docker/Dockerfile`; documentation string and whitespace checks. |
| Three dashboard names | Participant instructions consistently say click **Benchmarking** in the top menu. The visible page and PDF now use **Benchmarking** / **Benchmarking Report**. | Updated README, lesson, slide sources, and page/report headings. |
| Expected early evaluation years appear as errors | Already corrected: evaluation truth is loaded separately when its year is absent from the selected climatology ensemble. Such a year is evaluated rather than incorrectly skipped; genuinely missing observation files get a skip message. | Regression verifies separate target-year observation loading and absence of the former error messages. Lesson explains this distinction. |
| Onset Analysis reads nonexistent local CHIRPS folder | Fixed source selection to prefer the resolved observation directory/URI over the nominal `CHIRPS_025_DIR`. Pass that URI directly to ROMP's existing cloud-aware reader. The isolated worker preserves the container's bucket and authentication environment. | Reproduced three failing source/environment tests before fixing. A cloud-protocol fixture now reads real NetCDF rainfall, calculates onset/climatology, and saves a real PNG without the nominal local directory. |
| Status card is Idle during execution | Already corrected by active-job state taking precedence over the last completed run. | Strengthened Streamlit AppTest verifies the actual KPI HTML says Running, including after reconnect with an older completed result present. |
| Fresh-window report counts zero evaluated models | Fixed both report summary and PDF to derive deduplicated model rows from persisted scientific outputs, rather than current selections or session state. Baselines and empty/unrecognized artifacts are excluded. | Fresh-session integration with zero selected rows reports two stored models, excludes climatology, deduplicates windows, and generates a real PDF with matching total/deterministic/probabilistic counts. |
| No launch/dependency failures reported | Runtime interpreter selection and launch behavior retained. | Existing deterministic/probabilistic CLI/dashboard synthetic integration tests still pass. No dependencies added. |

## Report provenance

The report model list now reflects stored outputs. Current input controls may
describe a different experiment, so the UI and PDF explicitly distinguish those
controls from output provenance. Source-data inventory dates are not presented
as the historical evaluation dates of saved artifacts. Existing outputs are
counted without loading NetCDF scientific fields.

## Changed files

- `ui/pages/short_medium_benchmarking.py`: resolved Onset Analysis source, worker
  environment, consistent titles, and report integration.
- `ui/utils/benchmark_report.py`: persisted-result model inventory.
- `demos/demo4/README.md`, `docs/_episodes/04-weather-model.md`,
  `reports/slides/benchmarking.qmd`, `reports/slides/benchmarking_concerpt.qmd`:
  participant instructions and current-method clarification. Historical numerical
  tables remain unchanged. The two-window exercise explicitly disables dry
  confirmation rather than asking 45/46-day forecasts for 50 days of coverage.
- `tests/test_demo4_onset_inputs.py`, `tests/test_benchmark_report_models.py`,
  `tests/test_benchmark_reconnect.py`, `tests/test_romp_scoring_audit.py`:
  source/environment, real cloud-reader plotting, report/PDF, status, and log
  regressions.

The implementation reuses the existing cloud reader and serializers. It avoids
staging the complete observation archive for Onset Analysis and separates saved
evaluations from current model selection.

## Verification and limits

348 tests ran successfully, with one opt-in process-group test skipped. Python
compilation, targeted fatal/undefined-name Ruff checks, and whitespace checks
passed. The full suite includes real CLI/dashboard synthetic NetCDF comparisons,
the supplied scientific reference checks, a real generated PDF, Streamlit
interaction tests, and real onset plotting through a mocked bucket transport.

The mocked bucket supplies real NetCDF bytes through the production cloud reader;
this does not establish live production-bucket connectivity. The running Docker
deployment on port 8503 was not rebuilt/restarted, production forecasts were not
run, and the website/slides were not rendered during this follow-up. Rebuild the
image or reload the updated bind-mounted checkout before retesting that service.
