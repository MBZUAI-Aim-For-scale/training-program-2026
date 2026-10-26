# Demo 4: AI Weather Model Benchmarking

Because AI models and datasets are large, the benchmarking environment is containerized using Docker and hosted on a remote server. You will launch the backend via Docker and access the interactive Streamlit UI from your local machine using an SSH tunnel.

---

## Environment Setup

1. **SSH into the server**:
```bash
ssh <you>@country-spark

```


2. **Clone the repository and navigate to the project directory**:
```bash
git clone https://github.com/MBZUAI-Aim-For-scale/training-program-2026.git
cd training-program-2026

```


3. **Initialize environment configuration**:
```bash
cp .env.example .env
cat <<EOF > .env
DEMO_BUCKET=<bucket>/demo
DEMO3_BUCKET=<bucket>/demo3
DEMO4_BUCKET=<bucket>/demo4
EOF

```


4. **Start the container stack**:
```bash
bash docker/run.sh rebuild

```


*(For subsequent runs without rebuilding, use `bash docker/run.sh start` or `DEV=1 bash docker/run.sh start` for live code editing).*
5. **Create the SSH tunnel** (from your local machine):
```bash
ssh -L 8501:localhost:8501 -L 8888:localhost:8888 <you>@country-spark

```


6. **Access dashboards**:
  * **Streamlit UI**: `http://localhost:8501`
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




---

## UI-Based Execution (Streamlit Dashboard)

1. **Navigate to Benchmarking**: Open `http://localhost:8501` and select **Short Medium Benchmarking** in the sidebar.
2. **Configure Benchmark Inputs**:
  * **Ground Truth**: Select `CHIRPS` or `ENACTS`.
  * **Benchmark Coverage**: Select region (e.g., `Ethiopia`).
  * **Forecast Window**: Choose lead time (`1-15 days` or `16-30 days`).


3. **Select Models**: Filter by *Deterministic* or *Probabilistic* and select target models from the table.
4. **Adjust Run Windows & Shared Settings**:
  * Set Evaluation Start/End dates, Baseline Years, and Initialization Days (e.g., days 0, 3).
  * Configure event detection thresholds (e.g., Wet-day threshold = 20mm, Dry spell limit = 7 days).


5. **Execute Benchmark**: Click **Run Benchmark** to initiate `run_momp.py` with live logging.

---

## Exploring Results & Exporting

1. **Metric Map**: 2D spatial cross-sections (False Alarm Rate, Miss Rate, MAE) with skill thresholding, map subtraction, and swipe comparisons.
2. **All Metrics**:
* Summary evaluation matrix comparing model scores against Traditional Climatology.
* **Detailed Metric Distributions**: Grouped statistic cards (Mean, Min, Median, P75, P90, Max) highlighting best values in **Bold** and comparing model performance against climatology baselines.


3. **Benchmark Outputs**: Browse generated PNG output figures and inspect underlying NetCDF variables via the NetCDF Data Explorer.
4. **Onset Analysis**: Interactive time-series onset analysis with automatic grid snapping by coordinates or administrative Woredas.
5. **Skill vs Baseline & Uncertainty**: Verification scores relative to baseline, along with bootstrap confidence intervals and statistical significance.
6. **Benchmark Report & Export**: Generate and download a consolidated PDF report containing executive summaries, model inventory tables, and output figures.
