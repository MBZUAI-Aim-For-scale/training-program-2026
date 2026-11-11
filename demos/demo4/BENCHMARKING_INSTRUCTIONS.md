# Participant instructions: Run the benchmark

Use the **Benchmarking** dashboard. Start with the dashboard open and the training data available.

1. **Open Benchmarking** from the top menu.
2. **Choose your inputs** under **Benchmark Inputs**:
   - **Ground truth**: select the observational dataset assigned by your instructor.
   - **Benchmark coverage**: select your assigned country.
   - **Forecast window**: select **1-15 days** or **16-30 days**, as assigned.
3. **Select models** by clicking their rows in the **Models** table. Use **Filter by category** if needed. Wait for the selected models' data checks to finish.
4. **Check the run settings** for each selected model:
   - Set **Evaluation Start** and **Evaluation End** to the assigned evaluation period, within the available data range.
   - Set **Baseline Start Year** and **Baseline End Year** to the assigned climatology period.
   - Use the assigned **Initialization schedule** and **Verification windows**. Keep the same evaluation period, baseline years, and verification windows across models for comparison.
   - Leave other settings at their defaults unless instructed otherwise. Keep **Dry spell search extension** at **0**. For countries outside Ethiopia, keep **Area mask file** at **No mask** unless a country-specific mask is supplied.
5. **Run the benchmark**: resolve any validation messages, then click **Run Benchmark** once. Monitor the progress and live log; allow several minutes per model. Keep the dashboard open while it runs. Use **Stop Benchmark** only if you need to cancel.
6. **Confirm completion**: wait for **Latest benchmark run completed successfully.** If another forecast window is assigned, select it, check the verification windows, and repeat steps 4–6.

If you cannot start or finish:

- **Run Benchmark is disabled**: select at least one model, wait for data checks, and address the messages above the button.
- **Input files were added or replaced**: click **Refresh model data**, then recheck your selections.
- **The run fails**: open **Latest benchmark execution log** and share the error with your instructor.

Changing settings does not recompute saved results; click **Run Benchmark** to apply them.

## Practice setup

Use **Evaluation Start = 1 May 2020**, **Evaluation End = 31 July 2020**, **Baseline Start Year = 2018**, and **Baseline End Year = 2021**, or the facilitator's available equivalent including the evaluation year.

Follow the page order: **Benchmark Inputs → Models → Shared settings → Run Benchmark → Show saved results**. Under **Shared settings**, dates and baseline years appear beside the per-model **ADVANCED TEMPORAL & VERIFICATION SETTINGS**.

For the starting run, select only **AIFS Single**, **Forecast window = 1-15 days**, **Verification windows = Days 1–15**, and **Tolerance: days 1–15 = 3**. Set **Wet-day threshold = 20**, **Dry spell limit = 7**, **Dry spell search extension = 0**, and **Baseline forecast = climatology**. Keep the assigned initialization schedule, mask, wet-spell length, and remaining settings fixed.

Before changing settings, record the scores and settings or save screenshots. Later runs may replace earlier outputs. After each successful run, turn on **Show saved results**; use **Refresh saved results** if needed.

| Exercise | Change / question |
|:---|:---|
| 1. Establish a baseline | AIFS Single, Days 1–15, tolerance 3 days |
| 2. Rainfall threshold | Wet-day threshold: 20 → 30 mm |
| 3. Timing tolerance | Days 1–15 tolerance: 3 → 5 days |
| 4. Dry-spell check | Search extension: 21 → 0 days, if forecast coverage permits |
| 5. Ensemble reference | AIFS Ensemble; baseline start: 2018 → 2010, if available |
| 6. Lead windows and a finding | Both windows with equal 3-day tolerances |

## Exercise 1: Establish a baseline

1. Apply the practice setup and click **Run Benchmark**. Wait for successful completion.
2. Open **Show saved results → All Metrics**. Set **Metric model category → All**, untick **All models**, and select **AIFS Single** and **Traditional Climatology** in **Models to compare**.
3. Select **Lead times → Days 1–15** and record the values with their displayed units. Use **N/A** for unavailable values.

| Metric | AIFS Single | Traditional Climatology |
|:---|:---:|:---:|
| MAE (days) | | |
| Miss rate | | |
| False-alarm metric | | |

Record the observation years, actual initialization dates or schedule, baseline years, and reference settings used. Which values can you confirm from the execution log?

## Exercise 2: Raise the rainfall threshold

1. From the practice setup, change only **Wet-day threshold: 20 → 30** under **EVENT DETECTION**.
2. Predict which wet spells will still qualify as onset, then run the benchmark.
3. Record MAE, miss rate, and the false-alarm metric using the same **All Metrics** filters. Compare **Metric Map** views with the same model, window, and area.
4. Did all scores improve together? Restore **Wet-day threshold = 20**.

Despite its label, this control changes the **wet-spell rainfall total** used for observations, forecasts, and the reference. A different event definition does not demonstrate improved rainfall forecasts.

## Exercise 3: Change the timing tolerance

1. Restore the practice setup. In the selected model's advanced settings, change only **Tolerance: days 1–15: 3 → 5**.
2. Run and compare the same scores and maps with Exercise 1.
3. How does accepting a wider timing error change the scores? Restore the tolerance to **3**.

## Exercise 4: Switch off the dry-spell check

1. Restore the practice setup, then set **Dry spell search extension = 21**, keeping **Dry spell limit = 7**.
2. Run and record the scores **only if the data checks confirm enough forecast coverage**. A positive extension needs rainfall beyond the verification window; if unavailable, record **N/A** and skip this comparison.
3. Change only **Dry spell search extension: 21 → 0**, rerun, and compare the same scores and map views.
4. Could an early wet spell followed by dry weather now qualify? Would that definition suit planting advice?

Zero disables the follow-up dry-spell veto. Onset dates can change in forecasts, observations, and climatology; use your results to discuss possible false planting signals. Leave the extension at **0** for the remaining exercises.

## Exercise 5: Change the ensemble reference period

1. Restore the practice setup, then select only **AIFS Ensemble**, if available. Check its dates and advanced settings, run with baseline **2018–2021**, and record its available probabilistic scores and reference scores in **All Metrics**.
2. Change only **Baseline Start Year: 2018 → 2010**, keeping **Baseline End Year = 2021**. Run only if the observation years are available; otherwise record **N/A**.
3. How do the reference and skill scores change when the climatology period changes? Restore the baseline start to **2018**.

## Exercise 6: Compare lead windows and report a finding

1. Restore the AIFS Single practice setup. In advanced settings, select **Verification windows → Both windows**.
2. Set **Tolerance: days 1–15 = 3** and **Tolerance: days 16–30 = 3**. **Max Forecast Day (automatic)** becomes **30**; run only if the model supports this horizon.
3. In **All Metrics → Lead times**, select both windows. Compare the same metrics and area, keeping all other settings fixed.
4. Write one finding supported by your recorded values: which window performed better, on which metric, and under which settings? Note missing data or unequal sample coverage.

## Benchmark time limit

Dashboard benchmarks allow up to six hours per model. Exit code 124 means the
worker exceeded its time limit. For longer ensemble evaluations, set
`ROMP_BENCHMARK_TIMEOUT_SECONDS=43200` (12 hours) in the environment before
starting the dashboard. The value must be a positive integer in seconds.
Restart the dashboard after changing it, then rerun the failed model. You can
also reduce the evaluation years or initialization dates. The Stop button
remains available during long runs.
