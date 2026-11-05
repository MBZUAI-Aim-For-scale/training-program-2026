# Dashboard rerun and background-task investigation

Streamlit reruns are expected. The problem was the work reached during a rerun,
not top-to-bottom execution itself.

## Evidence and execution trace

Before the fix, `ui/app.py:main` loaded styles, called `pre_import_modules`
(importing every page, including scientific/native libraries), rendered navigation,
then called `collect_config_problems` for **every route** before rendering the
selected page. That check included remote existence checks and model directory
listings. Its 30-second cache only delayed recurrence; it did not isolate menus.
An AppTest regression reproduced Home importing all five menu modules. After the
fix, the same test visits every menu twice and verifies only its module is
imported and rendered.

Within the selected pages:

- Forecasting: `runs.tick` downloaded cloud files synchronously; changing controls
  replaced state even when a GPU task was running. Hidden result/model tabs still
  opened datasets and serialized downloads. Onset movie generation also ran on
  the UI thread. Cloud plan discovery and initial metadata checks could block on
  network availability.
- Benchmarking: model inventory and temporal validation could open remote files
  during UI construction. The onset button synchronously waited on a subprocess
  for up to 900 seconds. Preparing the onset year list could download the whole
  observation archive. Existing benchmark execution already used a background
  worker and cancellation; those semantics are preserved.
- Blending: input inspection ran synchronously; hidden result tabs loaded maps and
  download bytes. Existing pipeline execution was already asynchronous.
- Messages: authentication and district retrieval ran synchronously; district
  retrieval repeated on slider and download reruns. Forecast requests then waited
  up to 60 seconds **per district** on the script thread.

No infinite rerun loop was established. Finite operations with long network or
subprocess waits explain the unresponsive script thread. Ordinary finite loops
and dependency ordering should be retained.

## Changes

The app shell now dispatches only the selected page. Benchmark configuration
checks are an explicit background action within that menu. A shared status fragment
reads existing task snapshots without executing unrelated page logic.

`BackgroundTask` snapshots submission arguments, starts once, runs without any
Streamlit calls, and exposes a locked result/error snapshot. Session-owned polling
fragments apply completed results on the script thread. Failures stay visible;
submissions are disabled or rejected while their task is active. Inventory and
validation results use parameter-specific session caches. Lazy tab guards execute
only the selected tab's body, using Streamlit's supported `on_change="rerun"` and
`.open` API.

Independent menus and independent downloads can overlap. Forecast GPU queues retain
sequential execution and active state across changed controls. A bounded download
pool deduplicates writes to the same destination. Onset rendering uses isolated
processes, avoiding shared Matplotlib state. Benchmark/blending jobs sharing an
output directory serialize within the server process; waiting is cancel-aware.
Raw blending NetCDF reads use xarray's native-library lock, and observation cache
files are published atomically.

## Verification and limits

Run the existing suite with:

```sh
.venv/bin/python -m unittest discover -s tests
```

The responsiveness tests cover repeated menu navigation, blocked network work with
continued widget interaction, navigating away and back during a running task,
completion/error reporting, preserved settings/results, duplicate prevention,
closed result tabs, safe concurrent downloads, sequential GPU dependencies,
output-directory serialization and cancellation while waiting.

No production API forecasts, GPU inference, or full weather benchmark was launched
as part of verification. Slow external operations are exercised with deterministic
blocked workers and synthetic data. Session jobs survive ordinary reruns and menu
navigation; they are not a durable job queue across server restarts. Output locking
for benchmark/blending is scoped to a single server process; deployments with
multiple independent Streamlit server processes need external coordination.

Streamlit's supported APIs are documented at:
- https://docs.streamlit.io/develop/api-reference/layout/st.tabs
- https://docs.streamlit.io/develop/concepts/design/multithreading
- https://docs.streamlit.io/develop/concepts/architecture/fragments

Final verification: 212 tests ran successfully (one opt-in process-group test
skipped). Python compilation and isolated Ruff fatal/undefined-name checks passed
for the changed code. The repository-wide Ruff invocation is blocked by the
pre-existing underscore-form isort settings in `pyproject.toml`, so the targeted
check used `--isolated --select E9,F63,F7,F82`. The changed tracked code passes
`git diff --check`; an unrelated existing documentation EOF warning remains.

## Reconnecting benchmarks after browser refresh

Benchmark jobs now live in a locked server-side registry
(`ai_weather/benchmark_registry.py`). An unguessable identifier in the page's
`benchmark_job` query parameter reconnects a fresh Streamlit session to that same
job through `ui/utils/benchmark_session.py`; reconnecting does not call `start`.
Completed jobs remain available until 24 hours without access, with expired
completed handles pruned on registry access. Running handles are retained.

The app shell restores the handle before showing background status. The benchmark
page renders progress, model position, and Stop before inventory discovery can
return early, and its status card reflects the active job. Completion is applied
once per session, preventing repeated attachment/rerun loops. A still-open page's
existing job receives a reconnect identifier on rerun as well. Unknown/expired
identifiers show a clear message instead of attaching another job.

`tests/test_benchmark_reconnect.py` exercises fresh sessions, completed and expired
jobs, unknown identifiers, and existing-session migration. A Streamlit AppTest
creates two fresh browser sessions while the actual worker is blocked, verifies
the same log/model progress and exactly one runner invocation, and cancels that
same job through the refreshed page. The settings-state tests were adjusted for
completion polling at the beginning of the subsequent rerun.

This restores benchmark execution and progress within the same dashboard server
process. It does not persist jobs through a server restart or reconnect across
independent server processes. Other menus' background jobs retain their existing
session lifetime.

Refresh verification: 302 tests ran successfully (one opt-in test skipped),
including the fresh-session progress/cancellation checks and CLI/dashboard
scientific integration tests. Compilation, targeted fatal/undefined-name lint,
and whitespace checks passed for this change.
