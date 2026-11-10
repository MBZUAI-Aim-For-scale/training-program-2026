---
title: "Demo 3"
order: 3
teaching: 45
exercises: 30
questions:
- "How do we run AI weather models locally in a container?"
- "What are the core commands needed to generate a forecast figure?"
- "How do we tailor the results to specific use cases like rainy-season onset, heavy rain or heat?"
objectives:
- "Gain hands-on experience with the end-to-end process of running AI weather models locally."
- "Run one or several models from the Streamlit page and get use-case-specific outputs."
- "Interpret model outputs and compare forecasts against observations."
keypoints:
- "On a set-up Spark, the Demo 1 dashboard's Launch starts Demo 3; by hand it is build the image, start the container, open the page."
- "The use case chosen on the page (Heat, Precipitation or Onset) tailors the outputs: event movies, onset maps and the observations they are compared with."
- "Streamlit provides an interactive environment for executing and visualizing the AI weather models."
- "All configuration options are documented in Demo 1."
---

# Running AI Weather Forecast Models on a DGX Spark

This module provides a comprehensive guide to executing and benchmarking modern AI weather forecasting models using containerized workflows on a DGX Spark infrastructure. The Demo 3 Streamlit application serves as the unified interface for orchestrating seven AI weather models.

> #### Prerequisites
> 
> Complete the environment setup in [Demo 1: Setting Up AI Weather Forecasting Lab](/training-program-2026/01-setup) before proceeding. All configuration options (runtime flags, model environments, FGN setup, troubleshooting) are documented there.
{: .prereq}

## Learning Objectives

By the end of this lesson, you will be able to:

- Deploy and launch the Demo 3 Streamlit application on a DGX Spark.
- Execute AI weather models (AIFS, GraphCast, Aurora, Atlas CRPS, FGN, NeuralGCM) in containerized environments.
- Generate use-case-specific forecasts (temperature exceedance, precipitation exceedance, rainy-season onset).
- Compare model forecasts against observations (CHIRPS v3 rainfall, CHIRTS maximum temperature) using event movies.
- Interpret model performance metrics and output formats.

## Lesson Roadmap

| Section | Purpose |
| --- | --- |
| Application Overview | Understanding the Demo 3 architecture and capabilities |
| Quick Start | Opening Demo 3 from the Demo 1 dashboard |
| Model Execution | Running forecasts with the Streamlit interface |
| Output Interpretation | Understanding NetCDF outputs and visualization |
| Use-Case Customization | Tailoring models for specific forecasting challenges |
| Model Integration | Adding new models to the framework |
| Performance Benchmarks | Comparing model execution times and resource usage |
| Exercises | Hands-on practice with model execution |
| Extra: Run Demo 3 from the Terminal | Starting Demo 3 without the dashboard |

## Application Overview

### System Architecture

The Demo 3 application consists of:

- **Streamlit UI**: Interactive web interface for model selection and parameterization
- **Docker Container**: Isolated environment with all dependencies
- **Model Runners**: Specialized execution scripts for each AI model
- **Event Movie Pipeline**: Visualization comparing forecasts to observations
- **Output Storage**: NetCDF files with standardized data contracts

### Supported Models

| Model | Type | Resolution | Ensemble Members | Key Characteristics |
| --- | --- | --- | --- | --- |
| **AIFS v2 single** | Deterministic | 0.25° | 1 | ECMWF operational, fast inference |
| **AIFS v2 ENS** | Probabilistic | 0.25° | 2 by default | Ensemble forecasts, uncertainty quantification |
| **GraphCast** | Deterministic | 0.25° | 1 | Google DeepMind, graph neural network |
| **NeuralGCM** | Hybrid | 2.8° | 2 by default | Google, combines ML with physics |
| **Aurora 1.5** | Foundation | 0.25° | 1 | Microsoft, multi-variable Earth system |
| **Atlas CRPS** | Probabilistic | 0.25° | 2 by default | NVIDIA, noise-conditioned transformers |
| **FGN Mini** | Probabilistic | 1° | 3 by default | Google WeatherNext 2, cyclone tracking |

> #### Model Selection Guidance
> 
> - **Fast exploration**: AIFS v2 single (~2 min), FGN Mini (~2 min), GraphCast (~3.5 min)
> - **Probabilistic forecasting**: AIFS v2 ENS, Atlas CRPS, NeuralGCM, FGN Mini (you choose the number of members)
> - **Slow on the Spark**: Aurora 1.5 (~26 min) and Atlas CRPS (~19 min); in class, load their saved runs
> - **No 2 m temperature**: NeuralGCM, so it has no heat results
{: .tip}

### Current Status

- All 7 models run live on the Spark from one Docker image, and saved runs load instantly.
- Up to four models at once: saved runs load (or download from the shared bucket), the rest run one after another.
- Event movies (forecast vs observations) show side-by-side comparisons day by day; the Onset use case shows onset maps and movies.
- Every model's weights are downloaded during setup, so a model's first run doesn't wait for them.
- Results include execution telemetry, geospatial maps (z500, 2m temperature, rainfall), and downloadable NetCDF artifacts.
- Each model card transparently reports hardware requirements: weight size, peak GPU memory, output dimensions, and required input variables.

## 1. Quick Start

Start Demo 3 from the Demo 1 dashboard. The Spark's setup has already built Demo 3 and downloaded the models' weights, so there is nothing to install.

1. Open the dashboard at **http://localhost:8500** in the Spark's web browser (see [Access the Dashboard](/training-program-2026/01-setup/#access-the-dashboard) in Demo 1).
2. Click **3 · Run AI Forecast** in the top bar.
3. Click **Launch ↗**. A new browser tab opens, shows the start-up progress, then shows Demo 3 at **http://localhost:8502**.

If Demo 3 is already running, you can open **http://localhost:8502** directly.

> #### One demo at a time
>
> The demos share the Spark's GPU and memory. If another demo is running, the dashboard asks before stopping it, so check with your group first.
{: .callout}

To start Demo 3 from the terminal instead, see [Extra: Run Demo 3 from the Terminal](#extra-run-demo-3-from-the-terminal) at the end of this lesson.

## 2. Model Execution

### 2.1 Interactive Workflow

1. **Models**: pick up to four. Hover over a model's card for its Spark runtime and what it lacks (e.g., NeuralGCM has no 2 m temperature); **Model details** shows what each needs on the Spark.
2. **Forecast setup**: the start date (only dates every chosen model can start from are offered), the hour (00Z or 12Z), the lead time (in whole days, typed into the box, with no upper limit; past 15 days it warns that the models may lose skill) and, for ensemble models, the number of members.
3. **Region and use case**: a country (or a custom box), then **Heat**, **Precipitation** or **Onset**.
4. **Run**: one button loads the saved runs (from this Spark or the shared bucket) and runs the rest on the Spark one after another; its label says which, e.g. *Load 1 saved run and run 2 models one after another*. A table shows each model's progress.
5. **Results**: how each forecast was made (load, inputs, run, save, with times), event movies against observations (or, for Onset, onset maps and movies), a time-series comparison of every loaded model, and forecast maps with a NetCDF download.

The sidebar (closed by default: the arrow at the top left opens it) has developer options: the output folder, the shared bucket and the saved runs on this Spark.

### 2.2 Use-Case Group Execution

Organize into your designated use-case groups to begin the hands-on exercises. Each team picks its assigned challenge as the **Use case** on the page (Heat, Precipitation or Onset):

- **Temperature Exceedance**: Led by Docko, supported by Narayana.
- **Precipitation Exceedance (Short-run rainfall)**: Led by Koomi, supported by Shruti.
- **Onset/Cessation**: Led by Aryan, supported by Panchali.

> #### Exercise 1: Run Your First Forecast (15 min)
> 
> 1. Select the **AIFS v2 single** model.
> 2. Choose a start date within the model's valid range, at 00Z (before 13 May 2026 it can only start at 00Z).
> 3. Set lead time to 5 days.
> 4. Select your country and use case.
> 5. Run the forecast and explore the results.
> 
> **Questions:**
> - How long did the forecast take to run?
> - What variables are shown in the output maps?
> - How does the forecast compare to the observations in the event movie?
{: .exercise}

<details>
<summary><strong>Discussion Points</strong></summary>
<br>

> #### Solution
> 
> - AIFS v2 single takes about 2 minutes for a 10-day forecast, so less for 5 days (the run's times are under **How this forecast was made**).
> - Output includes z500 (geopotential), t2m (2m temperature), and tp (precipitation).
> - The event movie shows day-by-day comparison; look for systematic biases (e.g., temperature too warm or cool in specific regions).
{: .solution}

</details>

## 3. Output Interpretation

### 3.1 Data Contract

Every model execution produces a single, globally covered NetCDF file. All runners must strictly adhere to this schema:

| Attribute | Specification |
| --- | --- |
| **dims** | `ensemble, lead_time, lat, lon` (`ensemble` dimension size is 1 for deterministic models) |
| **`lead_time`** | Integer hours post-initialization: 0, 6, 12, … |
| **`lat` / `lon`** | Ascending order; `lon` spans −180 to 180 |
| **`tp`** | Total precipitation accumulated over the preceding 6 hours, in **mm** (NaN at lead 0) |
| **`t2m`** | 2-meter temperature, in **K** |
| **`z500`** | 500 hPa geopotential, in **m² s⁻²** (divide by 9.80665 to derive geopotential height in meters) |
| **coords** | `valid_time` (aligned with `lead_time`), `init_time` |
| **attrs** | `model`, `model_name`, `init_source`, `members` |

### 3.2 Loading Outputs in Python

```python
import xarray as xr

# Load a forecast
ds = xr.open_dataset("outputs/aifs2_single/20250703T00_240h_m1.nc")

# Access 2m temperature at lead time 24h
t2m_24h = ds["t2m"].sel(lead_time=24)

# Convert geopotential to height (meters)
z500_height = ds["z500"] / 9.80665

print(ds)
```

> #### Output Format Essentials
> 
> - All models produce the same NetCDF structure for consistency.
> - Precipitation is accumulated over 6-hour windows.
> - Geopotential must be divided by gravity (9.80665) to get height in meters.
> - The ensemble dimension has size 1 for deterministic models.
{: .keypoints}

### 3.3 Event Movie Pipeline

The visualization pipeline reads Demo 3 output files and the observations: from this Spark's country data first (every country downloaded from the Demo 1 dashboard), else from the program's bucket. The selected use case dictates the movie variable and the observations:

- **Heat**: daily maximum 2m temperature, against CHIRTS (ERA5 for regions outside the programme countries). The models output every 6 hours, so their daily maximum reads about 0.3–1.3 °C low.
- **Precipitation**: daily rainfall accumulation, against CHIRPS v3.

Movies are rendered on-demand and cached in `outputs/_movies/`.

## 4. Use-Case Customization

### 4.1 Temperature Exceedance

Focus: Heatwave tracking and daily maximum 2m temperature thresholds.

**Key parameters:**
- Variable: `t2m` (2-meter temperature)
- Metric: Daily maximum temperature; the regional series is the area maximum by default (or the mean, 95th or 99th percentile)
- Threshold: automatic (the 95th percentile of the observations), one of the program's thresholds (37–45 °C), or one you type

### 4.2 Precipitation Exceedance

Focus: Short-run rainfall accumulation and flood risk mapping.

**Key parameters:**
- Variable: `tp` (total precipitation)
- Metric: Daily rainfall; the regional series is the area mean by default (or the maximum, 95th or 99th percentile)
- Threshold: automatic (the 97.5th percentile of the observations), one of the program's thresholds (1–60 mm/day), or one you type

### 4.3 Onset

Focus: when the rainy season starts, at every grid point.

**Key parameters:**
- Variable: `tp` (precipitation): observed CHIRPS v3 up to the day before the forecast, then the forecast
- Definition: ICPAC (the default in most countries), Moron–Robertson, or Moron–Robertson with the MOK filter (the default for India, Bangladesh and Pakistan); every threshold can be changed on the page
- Metric: wet-spell and dry-spell detection (see [Demo 5](/training-program-2026/04-weather-model) for full ROMP details). A wet spell whose dry-spell check runs past the end of the forecast shows as *possible* onset
- Application: Rainy season onset timing

> #### Use-Case Flags
> 
> Each use case activates specific visualization pipelines and metric calculations. The Streamlit interface automatically adjusts the output based on your selection.
{: .tip}

## 5. Model Integration

### 5.1 Adding a New Model

To integrate a new AI weather model into the framework:

1. **Create runner script**: `runners/runner_<model_key>.py`

   ```python
   def run(init, lead_hours, members, report) -> xr.Dataset:
       # Generate forecast
       # Return Dataset with tp, t2m, z500 in the units specified above
       pass
   ```

2. **Provision environment**: Ensure the model's Python environment exists at `/opt/envs/<env>/bin/python` inside the container (see [Demo 1](/training-program-2026/01-setup) for environment details).

3. **Register model**: Add its entry to `MODELS` in `demo3/catalog.py`; it runs live once its environment and runner exist.

4. **Weights**: Teach `runners/prefetch_weights.py` where its weights come from, so setup downloads them ahead.

5. **Benchmark**: Add the measured runtime to `timings.json`.

> #### Data Contract Compliance
> 
> New models must strictly adhere to the output format (tp in mm, t2m in K, z500 in m² s⁻²). Non-compliant outputs will break the visualization pipelines.
{: .warning}

### 5.2 Core Directory Layout

*(Paths are relative to `training-program-2026/demos/demo3/`)*

```text
app.py                 # Main Streamlit application: orchestration and results display
demo3/catalog.py       # Model registry (type, environment, lead-time limits, member counts)
demo3/regions.py       # Geographic mapping: program countries to padded plotting bounding boxes
demo3/contract.py      # Strict data contract: output-file format enforced for all runners
demo3/store.py         # Artifact management: storage paths and saved-run deduplication
demo3/runs.py          # Gets the selected models' forecasts: saved, downloaded, or run one at a time
demo3/jobs.py          # Asynchronous job launcher and progress tracker
demo3/movie.py         # Background event movie generator for loaded runs
demo3/results.py       # Pipeline visualization: model requirements, forecast maps, NetCDF export
demo3/compare.py       # Time-series comparison of the loaded models
demo3/obs.py           # Observations: CHIRPS v3 rainfall and CHIRTS maximum temperature
demo3/onset.py         # Onset definitions and detection (matches ROMP)
demo3/onset_view.py    # The Onset results: definition picker, maps, series, onset movie
demo3/event_movie/     # forecast_event_movie.py + Spark adaptation layer
demo3/theme.py         # UI styling, aligned with the Demo 5 platform design system
runners/run_model.py   # Universal runner entry point
runners/fgn_convert.py # FGN-specific starting condition converter from ECMWF open data
runners/prefetch_weights.py  # Downloads every model's weights ahead (docker/prefetch.sh)
timings.json           # Empirically measured Spark runtimes displayed in the UI
docker/                # Containerization: Dockerfile, run.sh, prefetch.sh, per-environment dependency locks
outputs/               # Persistent storage: outputs/{model}/{YYYYMMDDTHH}_{lead}h_m{members}.nc
```

## 6. Performance Benchmarks

### 6.1 Execution Times (10-day forecast)

| Model | Members | Execution Time | Peak GPU Memory |
| --- | --- | --- | --- |
| **AIFS v2 single** | 1 | ~2 min | 14 GB |
| **FGN Mini (1°)** | 3 | ~2 min* | 1.8 GB |
| **NeuralGCM (2.8°)** | 3 | ~2.5 min | 18 GB |
| **GraphCast** | 1 | ~3.5 min | 16 GB |
| **AIFS v2 ENS** | 3 | ~5.5 min | 25 GB |
| **Atlas CRPS** | 3 | ~19 min | 33 GB |
| **Aurora 1.5** | 1 | ~26 min | 27 GB |

\* **FGN Note:** Time is extrapolated from a measured 106 seconds for Google's 7.5-day sample case.

> #### Performance Trade-offs
> 
> - **Time step**: Aurora 1.5 steps one hour at a time (240 steps for 10 days), the others six hours, which is much of why it is slow.
> - **Resolution vs. Memory**: Higher resolution (0.25°) requires more GPU memory.
> - **Deterministic vs. Probabilistic**: Ensemble models provide uncertainty estimates but take longer.
{: .tip}

### 6.2 Initialization Overhead

Each model's weights (~21 GB for all seven) download once into `~/.cache/demo3`. The Spark's setup downloads them in the background after building the image (`docker logs -f demo3-weights` shows how far it has got; `bash docker/prefetch.sh` does it by hand), so a model's first run in class doesn't wait for them: without that, Atlas CRPS's 12.5 GB alone adds about 31 minutes to its first run. Loading cached weights takes seconds (4–41 s).

### 6.3 FGN (WeatherNext 2) Notes

FGN is Google DeepMind's Functional Generative Network for probabilistic forecasting. It requires special handling due to its unique data requirements. Detailed configuration is in [Demo 1, Section 6](/training-program-2026/01-setup).

- **Model variant**: `WeatherNextCyclones_Mini` (1° resolution, the only Mini weights published)
- **Starting conditions**: Google's sample file for 2024-10-07 (up to 7.5 days); converter for all other dates
- **Performance**: 10-day, 3-member forecast in ~2 minutes, 1.8 GB peak memory
- **Input cache**: 53 MB per start date, reused for subsequent runs

## Summary

- Demo 3 provides a unified interface for seven AI weather models.
- Containerized deployment ensures reproducibility across DGX Spark nodes.
- On a set-up Spark, the Demo 1 dashboard's Launch starts it; the models' weights are already downloaded.
- The use case (Heat, Precipitation, Onset) tailors the outputs and the observations they are compared with.
- All models produce standardized NetCDF outputs for consistent analysis.
- Performance varies from 2 minutes (AIFS) to 26 minutes (Aurora) for 10-day forecasts.
- All configuration options are documented in [Demo 1](/training-program-2026/01-setup).

## Exercises

> #### Exercise 2: Compare Deterministic vs. Probabilistic (20 min)
> 
> Run both **AIFS v2 single** (deterministic) and **AIFS v2 ENS** (probabilistic, 3 members) for the same start date and lead time.
> 
> **Questions:**
> 1. How do the execution times compare?
> 2. What additional information does the ensemble provide?
> 3. How would you use the ensemble spread to assess forecast confidence?
{: .exercise}

<details>
<summary><strong>Discussion Points</strong></summary>
<br>

> #### Solution
> 
> 1. AIFS v2 single: ~2 min; AIFS v2 ENS: ~5.5 min (2.75× slower for 3 members).
> 2. The ensemble provides uncertainty estimates; you can compute spread (standard deviation across members).
> 3. Large spread indicates low confidence; small spread suggests high confidence. Compare spread to historical verification to assess reliability.
{: .solution}

</details>

> #### Exercise 3: Interpret Model Outputs (15 min)
> 
> Load a forecast NetCDF file and answer the following:
> 
> ```python
> import xarray as xr
> ds = xr.open_dataset("outputs/graphcast/20250701T00_240h_m1.nc")
> ```
>
> (Load that saved run on the page first: GraphCast, 1 Jul 2025 00Z, 10 days.)
> 
> 1. What are the dimensions and coordinates?
> 2. Extract the 500 hPa geopotential height (in meters) at lead time 48h.
> 3. What is the total precipitation accumulated between lead times 24h and 48h?
{: .exercise}

<details>
<summary><strong>Check your answer</strong></summary>
<br>

> #### Solution
> 
> ```python
> # 1. Dimensions and coordinates
> print(ds.dims)   # {'ensemble': 1, 'lead_time': 41, 'lat': 721, 'lon': 1440}
> print(ds.coords)
> 
> # 2. Geopotential height at 48h
> z500_48h = ds["z500"].sel(lead_time=48)
> z500_height = z500_48h / 9.80665  # Convert to meters
> 
> # 3. Precipitation accumulation (24h to 48h)
> # tp at lead L is the rain in the 6 h before it, (L-6, L], so 24-48 h is leads 30 to 48
> tp_24_48 = ds["tp"].sel(lead_time=slice(30, 48)).sum(dim="lead_time")
> ```
{: .solution}

</details>

> #### Exercise 4: Use-Case Group Activity (30 min)
> 
> In your assigned use-case group:
> 
> 1. Run forecasts for your specific challenge (temperature, precipitation, or onset).
> 2. Generate event movies comparing forecasts to observations.
> 3. Identify one systematic bias in the model output.
> 4. Prepare a 2-minute presentation on your findings.
> 
> **Deliverables:**
> - Screenshot of the event movie
> - Description of the bias observed
> - Hypothesis for why the bias occurs
{: .exercise}

## Troubleshooting

> #### Complete Troubleshooting Guide
> 
> For comprehensive troubleshooting including GPU access, container conflicts, network issues, and model-specific problems, see [Demo 1: Troubleshooting Section](/training-program-2026/01-setup).
{: .tip}

| Symptom | Quick Resolution |
| --- | --- |
| Container won't start: "unresolvable CDI devices" | `systemctl --user restart docker && docker start demo3` |
| Port 8502 already in use | `docker ps`, then `docker stop <name>`: something else took Demo 3's port (each demo has its own, set in `demos/demo1/demos.toml`) |
| A model's first run downloads its weights | Setup downloads them in the background: `docker logs demo3-weights` shows how far it has got; `bash docker/prefetch.sh` starts it again |
| `Conflict. The container name "/demo3" is already in use` | `docker rm -f demo3`, then `bash docker/run.sh` (the latest `run.sh` does this itself) |
| Model run fails during download | ECMWF mirrors throttle at times; Demo 3 tries the other one by itself. If both fail, try again later |
| Out of memory errors | Check GPU memory with `nvidia-smi` and stop other demos' containers (`docker stop <name>`); members run one after another, so fewer members don't lower the peak |

## Extra: Run Demo 3 from the Terminal

You don't need this section in class: the dashboard's **Launch** button does all of it. It is here for running Demo 3 by hand, for example while developing it. For detailed setup (GPU verification, container building, runtime configuration flags, model environments), see [Demo 1](/training-program-2026/01-setup).

### Core Commands

```bash
# 1. Go to the demo (the code is already on the Spark; if it isn't, first run
#    git clone https://github.com/MBZUAI-Aim-For-scale/training-program-2026.git ~/training-program-2026)
cd ~/training-program-2026/demos/demo3

# 2. Build the container (first time only, up to an hour)
docker build -t demo3 -f docker/Dockerfile .

# 3. Start the container
bash docker/run.sh

# (optional) Download every model's weights (~21 GB) in the background, as setup does
bash docker/prefetch.sh

# 4. Open http://localhost:8502 in the Spark's web browser
```

Starting Demo 3 this way doesn't stop other demos for you. If another demo is running, stop it first (from the dashboard, or `docker stop <name>`), since they share the GPU and memory.

### Runtime Options

Common configuration flags (see [Demo 1](/training-program-2026/01-setup) for the complete list):

```bash
# Development mode (live code updates without rebuild)
DEV=1 bash docker/run.sh

# Custom cache location for model weights
DEMO3_CACHE=/shared/cache bash docker/run.sh

# Try the Azure mirror for ECMWF data first (the other mirror is tried if one fails)
DEMO3_IFS_SOURCE=azure bash docker/run.sh
```

> #### First-Run Downloads
>
> If you skip `bash docker/prefetch.sh`, each model downloads its weights the first time it runs (~21 GB for all seven). This adds about 31 minutes to Atlas CRPS's first run. Later runs start in seconds.
{: .warning}

## Next Steps

After completing this demo, proceed to **Demo 4** (AfriClimate AI ForecastCheck) to verify AI models over years of past forecasts, then **Demo 5** (Onset Benchmarking) to learn systematic benchmarking of AI weather models against local observations for rainy season onset detection.
