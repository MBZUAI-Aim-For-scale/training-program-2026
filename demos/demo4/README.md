# Demo 4 · Weather model scorecard
## Environment Setup

Use the Docker setup below for the full benchmark. It includes the Python environment needed by the CLI and dashboard.

The CHIRPS v3 and updated forecast country boxes use these paths directly under
`gs://<DEMO_BUCKET>` (without `DEMO_BUCKET_PREFIX`). The country comes from
`ROMP_REGION`, which defaults to `Ethiopia`.

| Dataset | Bucket-relative file path | Supplied years |
| --- | --- | --- |
| CHIRPS | `CHIRPSv3_Country_0p25deg/{Country}/CHIRPSv3_{Country}_{year}_box025.nc` | 2000– |
| AIFS | `Forecasts_Country_0p25deg/AIFS_v2/tp/{Country}/AIFS_v2_tp_{Country}_{year}_box025.nc` | 2000–2025 |
| AIFS Ensemble | `Forecasts_Country_0p25deg/AIFS_ENS_v2/tp/{Country}/AIFS_ENS_v2_tp_{Country}_{year}_box025.nc` | 2000–2025 |
| GraphCast | `Forecasts_Country_0p25deg/graphcast/tp/{Country}/graphcast_tp_{Country}_{year}_box025.nc` | 2000–2025 |

All four are on one 0.25° grid (cells on whole and quarter degrees; longitudes 0..360 for countries
west of Greenwich, as in the files), so observed and forecast onsets compare point for point. The
older `CHIRPSv3_Country_Boxes_025deg` files are on CHIRPS's own grid, half a cell off, and aren't used.

On a Spark, these files are read from its country data first (what Demo 1's country-data button
downloaded, at the bucket's paths; `docker/run.sh` mounts it read-only at `/country_data`), and from
the bucket otherwise. Listings (which years and models there are) still come from the bucket, so a
partial download can't hide any; only if the bucket can't be reached does the country data stand in.

Local fallback retains the existing model directories and `{year}.nc` files.

### Remote Server Setup with Docker
You will need two terminal windows: one connected to the remote server to run the Docker container, and one on your local machine to create the SSH tunnel.

1. **SSH into the Spark server** using your preferred terminal:
   ```bash
   ssh <you>@country-spark
   ```
2. **Navigate to the project directory** (if not already there).
3. **Start the container** by running the `run.sh` script. This builds the image, mounts the necessary cache volumes (for the model weights), and starts the Streamlit server on port 8503:
   ```bash
   bash demos/demo4/docker/run.sh rebuild
   ```

or

  ```bash
   docker restart ai_weather
   ```



<<<<<<< HEAD
   *(Tip: If you are actively editing the code and want changes to reflect live without rebuilding the image, use `DEV=1 bash demos/demo4/docker/run.sh` instead).*
=======
```


3. **Environment configuration (optional)**: without a `.env`, `docker/run.sh` uses the
program's bucket (`DEMO_BUCKET=aim4scale_training_2026`) and, on a Spark, its key at
`/etc/fleet-gcs-key.json`. To use another bucket:
```bash
cp .env.example .env    # then set DEMO_BUCKET=<bucket> in it
```


4. **Start the container stack**:
```bash
bash demos/demo4/docker/run.sh rebuild

```


*(For subsequent runs without rebuilding, use `bash demos/demo4/docker/run.sh start` or `DEV=1 bash demos/demo4/docker/run.sh start` for live code editing).*
5. **Create the SSH tunnel** (from your local machine):
```bash
ssh -L 8503:localhost:8503  <you>@country-spark

```


6. **Access dashboards**:
  * **Streamlit UI**: `http://localhost:8503`
  * **Jupyter Lab**: `http://localhost:8888`


## Command-Line Execution (`run_momp.py`)

The `run_momp.py` script is the execution engine. Execute commands inside the Docker container:

1. **Enter the Docker container**:
```bash
docker exec -it ai_weather bash

```


2. **Run a Deterministic Model** (e.g., GraphCast, FuXi, AIFS Single v2):
```bash
python run_momp.py --mode det --model GraphCast

```


3. **Run a Probabilistic (Ensemble) Model** (e.g., AIFS Ensemble v2, GenCast):
```bash
python run_momp.py --mode prob --model AIFS_ENS

```
>>>>>>> 3e0f7a15957d9bbcb53b0d3f64d9ab09dea9db9a

   *(Tip: If you are actively editing the code and want changes to reflect live without rebuilding the image, use `DEV=1 bash docker/run.sh` instead).*
>>>>>>> a3069a729169017a7198e049f26cbb2bd4bf3fc7
4. **Create the SSH Tunnel** (Local Machine): Open a new terminal window on your local laptop. Set up port forwarding so your local browser can access the remote Streamlit server:
   ```bash
   ssh -N -L 8503:127.0.0.1:8503 <you>@country-spark
   ```
   *(Leave this terminal open while you work).*
5. **Access the Dashboard**: Open your local web browser and navigate to `http://localhost:8503`.

The relocated Compose stack publishes remote host port **8503** to Streamlit's
container port **8501**. The SSH tunnel must target remote port **8503**. To keep
using `http://localhost:8501` locally, use
`ssh -N -L 8501:127.0.0.1:8503 <you>@country-spark` instead.

If SSH reports `connect: connection refused`, check the stack on the remote host:

```bash
docker compose --env-file .env -f demos/demo4/docker/docker-compose.yml ps
docker compose --env-file .env -f demos/demo4/docker/docker-compose.yml logs --tail=80 ai_weather
curl -f http://127.0.0.1:8503/_stcore/health
```

Run these commands from the repository root. A successful image build does not
start the server; `bash demos/demo4/docker/run.sh start` starts it and waits for
Streamlit's health endpoint. The script resolves its paths when called from
inside `demos/demo4/docker` too (`bash run.sh start`).



---

## Command-Line Execution (`run_momp.py`)

The `run_momp.py` script is the backend execution engine. It processes one model per run, categorizing them into Deterministic (`det`) or Probabilistic (`prob`) modes. If you are using the Docker setup, you must execute these commands inside the running container.

1. **Enter the Docker Container** (Remote Server Terminal):
   ```bash
   docker exec -it ai_weather bash
   ```
2. **Run a Deterministic Model** (e.g., GraphCast, FuXi, AIFS):
   Inside the container, `python` resolves to `/opt/envs/app/bin/python`, the application environment installed in the image. You do not need to create a host `.venv` for the Docker workflow. Only when working **outside Docker** in an already configured host development checkout should you replace `python` with `.venv/bin/python` in these commands.

   ```bash
   python run_momp.py --mode det --model GraphCast
   ```
3. **Run a Probabilistic (Ensemble) Model** (e.g., AIFS_ENS, GenCast):
   ```bash
   python run_momp.py --mode prob --model AIFS_ENS
   ```
4. **Run with a Benchmark Specification File**:
   If you have a predefined `benchmark_spec.json` containing evaluation windows and shared settings:
   ```bash
   python run_momp.py --mode det --model FuXi --spec path/to/benchmark_spec.json
   ```
5. **Exit the container** when finished:
   ```bash
   exit
   ```

---

## Part 3: UI-Based Execution (Streamlit Dashboard)

The interactive dashboard provides a visual interface to configure, execute, and analyze the benchmark without needing to type CLI commands.

1. **Open Benchmarking**: In your local browser at `http://localhost:8503`, click **Benchmarking** in the top menu. Use that menu rather than Streamlit's hidden sidebar.
2. **Configure Benchmark Inputs**:
   - **Ground Truth**: Select the observational dataset (e.g., CHIRPS or ENACTS).
   - **Benchmark Coverage**: Select the region (e.g., Ethiopia). Note: The run button will be disabled if the region's data or masks are not configured in `config.py`.
   - **Forecast Window**: Choose the lead time window (1-15 days or 16-30 days).
3. **Select Models**:
   - The UI auto-detects available models from your data directories.
   - Use the "Filter by category" radio buttons to view Deterministic or Probabilistic models.
   - Click the rows in the table to select the models you want to benchmark.
4. **Adjust Model Run Windows & Settings**:
   - Expand the model details to set the Evaluation Start/End dates and Baseline Years.
   - Select Initialization Days (e.g., day 0, day 3).
   - (Optional) Expand "Advanced Temporal & Verification Settings" to tweak tolerances and verification windows.
   - Scroll down to "Shared Settings" to adjust event detection thresholds (e.g., Wet-day threshold, Dry spell limit) and area masks.
   - The reference default **Dry spell search extension = 0** disables dry confirmation. A positive value changes the onset definition and requires additional forecast coverage; the page checks the selected model files before starting.
5. **Execute the Benchmark**:
   - Click the "Run Benchmark" button.
   - A live log stream will appear at the bottom of the page showing the progress of the `run_momp.py` subprocess running inside the Docker container.
   - Wait for the completion message. This can take several minutes per model depending on the server's hardware.

---

## Exploring Results & Exporting

After the benchmark finishes, enable **Show saved results** to load the Results tabs:

1. **Metric Map**: Visualize spatial metrics (False Alarm Rate, Miss Rate, MAE). You can swipe between models, compare lead times, and clip to specific administrative regions if a shapefile is configured.
2. **All Metrics**: View a comprehensive, color-coded HTML table comparing all models across all windows, alongside detailed distribution statistics (Mean, Min, Median, P75, P90, Max).
3. **Benchmark Outputs**: Browse the generated PNG figures categorized by model type (Deterministic, Probabilistic, Climatology) and explore the raw NetCDF datasets.
4. **Onset Analysis**: Run a localized time-series onset analysis for specific coordinates or administrative regions.
5. **Skill vs Baseline**: View verification scores comparing the AI models against the Traditional Climatology baseline.
6. **Uncertainty**: Compute and view bootstrap confidence intervals and statistical significance.
7. **Benchmark Report & Export**:
   - Click "Generate Complete PDF Report" to compile an executive summary, fair-comparison plan, verification scores, and all generated figures into a single scientific PDF.
   - Click "Download Complete PDF Report" to save it to your local machine.

Probabilistic runs with score saving enabled also write
`<model>/probabilistic_metrics_<model>_<start>-<end>.nc`. **All Metrics** uses these
per-cell Brier, RPS, skill, and AUC grids for detailed distribution statistics.
Each cell uses the same paired reference samples and complete RPS categories as
the pooled scoring pipeline. Cells with undefined AUC or zero reference-score
denominators remain missing. The summary CSV scores pool all cases, so they can
differ from averages of per-cell scores. Older CSV-only runs must be rerun to
generate the grids; refreshing the page cannot reconstruct them.

Ensemble runs also generate the MAE/FAR/MR panel portrait from consensus onset
dates, using the reference ROMP member-agreement threshold and rounded mean
onset timing. The panel compares those onset errors with climatology; FAR and
MR differences are shown in percentage points. It is saved in the forecast
model's figure folder as `panel_portrait_mae_far_mr_<model>_<horizon>day.png`.
Rerun an older ensemble benchmark to generate this plot.

Evaluation and climatology periods may differ. For example, evaluation years 2003–2006 can be verified against a 2007–2024 climatology by loading those evaluation years' observations separately; they are not errors merely because they fall outside the baseline period. If an observation file is actually missing, the log identifies the missing year and the resulting sample coverage must be checked.
