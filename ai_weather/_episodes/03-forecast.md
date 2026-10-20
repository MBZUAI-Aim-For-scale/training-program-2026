---
title: "Demo 3"
teaching: 45
exercises: 30
questions:
- "How do we run AI weather models locally in a container?"
- "What are the core commands needed to generate a forecast figure?"
- "How do we tailor the model execution to specific use cases like onset, cessation, or temperature exceedance?"
objectives:
- "Gain hands-on experience with the end-to-end process of running AI weather models locally."
- "Execute a streamlined 4-command workflow to run a model and generate use-case-specific outputs."
- "Interpret model outputs and compare forecasts against observations."
keypoints:
- "A streamlined 4-command workflow (build container, run model, get output, generate figure) simplifies local AI forecasting."
- "Use-case flags in the code allow groups to tailor outputs for onset/cessation, temperature exceedance, or precipitation exceedance."
- "Streamlit provides an interactive environment for executing and visualizing the AI weather models."
- "All configuration options are documented in Demo 1."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Running AI Weather Forecast Models on a DGX Spark

This module provides a comprehensive guide to executing and benchmarking modern AI weather forecasting models using containerized workflows on a DGX Spark infrastructure. The Demo 3 Streamlit application serves as the unified interface for orchestrating seven AI weather models.

> #### Prerequisites
> 
> Complete the environment setup in [Demo 1: Setting Up AI Weather Forecasting Lab](01-setup.md) before proceeding. All configuration options (runtime flags, model environments, FGN setup, troubleshooting) are documented there.
{: .prereq}

## Learning Objectives

By the end of this lesson, you will be able to:

- Deploy and launch the Demo 3 Streamlit application on a DGX Spark.
- Execute AI weather models (AIFS, GraphCast, Aurora, Atlas CRPS, FGN, NeuralGCM) in containerized environments.
- Generate use-case-specific forecasts (temperature exceedance, precipitation exceedance, onset/cessation).
- Compare model forecasts against ERA5 observations using event movies.
- Interpret model performance metrics and output formats.

## Lesson Roadmap

| Section | Purpose |
| --- | --- |
| Application Overview | Understanding the Demo 3 architecture and capabilities |
| Quick Start | Running your first forecast in four commands |
| Model Execution | Running forecasts with the Streamlit interface |
| Output Interpretation | Understanding NetCDF outputs and visualization |
| Use-Case Customization | Tailoring models for specific forecasting challenges |
| Model Integration | Adding new models to the framework |
| Performance Benchmarks | Comparing model execution times and resource usage |
| Exercises | Hands-on practice with model execution |

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
| **AIFS v2 ENS** | Probabilistic | 0.25° | 3–50 | Ensemble forecasts, uncertainty quantification |
| **GraphCast** | Deterministic | 0.25° | 1 | Google DeepMind, graph neural network |
| **NeuralGCM** | Hybrid | 2.8° | 3 | Google, combines ML with physics |
| **Aurora 1.5** | Foundation | 0.25° | 1 | Microsoft, multi-variable Earth system |
| **Atlas CRPS** | Probabilistic | 0.25° | 3 | NVIDIA, noise-conditioned transformers |
| **FGN Mini** | Probabilistic | 1° | 3 | Google WeatherNext 2, cyclone tracking |

> #### Model Selection Guidance
> 
> - **Fast exploration**: AIFS v2 single (~2 min), FGN Mini (~2 min)
> - **Probabilistic forecasting**: AIFS v2 ENS, Atlas CRPS
> - **High accuracy**: Aurora 1.5, GraphCast (but slower: 26 min and 3.5 min respectively)
{: .tip}

### Current Status

- All 7 models run live on the Spark from one Docker image, and saved runs load instantly.
- The event movie tab (forecast vs observations) renders side-by-side comparisons day by day.
- Results include execution telemetry, geospatial maps (z500, 2m temperature, rainfall), and downloadable NetCDF artifacts.
- Each model card transparently reports hardware requirements: weight size, peak GPU memory, output dimensions, and required input variables.

## 1. Quick Start

> #### Complete Setup Guide
> 
> For detailed setup instructions including GPU verification, container building, runtime configuration flags, model environments, and troubleshooting, see [Demo 1: Setting Up AI Weather Forecasting Lab](01-setup.md).
{: .tip}

### 1.1 Core Commands

```bash
# 1. Clone and navigate to the demo
git clone https://github.com/mesfind/ai_weather.git
cd ai_weather/demos/demo3

# 2. Build the container (first time only, ~1 hour)
docker build -t demo3 -f docker/Dockerfile .

# 3. Start the container
bash docker/run.sh

# 4. Access from your laptop (on Tailscale)
ssh -L 8501:localhost:8501 <user>@<spark>
# Then browse to http://localhost:8501
```

### 1.2 Runtime Options

Common configuration flags (see [Demo 1](01-setup.md) for the complete list):

```bash
# Development mode (live code updates without rebuild)
DEV=1 bash docker/run.sh

# Custom cache location for model weights
DEMO3_CACHE=/shared/cache bash docker/run.sh

# Use Azure mirror for ECMWF data
DEMO3_IFS_SOURCE=azure bash docker/run.sh
```

> #### First-Run Downloads
> 
> The first execution of each model downloads weights (~75 GB total). This can take 30+ minutes for large models like Atlas CRPS. Subsequent runs start in seconds. Pre-run heavy models before the session for a smooth demonstration.
{: .warning}

## 2. Model Execution

### 2.1 Interactive Workflow

1. **Model Selection**: Choose **Deterministic** or **Probabilistic**, then select a specific model. The UI card displays benchmarked Spark runtimes and known limitations (e.g., NeuralGCM lacks 2m temperature outputs).
2. **Parameterization**: Set the start date (validated against the model's training domain), lead time, ensemble members, geographic region, and use case.
3. **Execution**: Click **Run forecast** (or **Load saved run** if a matching forecast already exists).
4. **Analysis**: Explore the result tabs, including the **Event movie** tab for temporal forecast-to-observation comparisons.

### 2.2 Use-Case Group Execution

Organize into your designated use-case groups to begin the hands-on exercises. Ensure your team uses the specific flag in the code that tailors the model output to your assigned challenge:

- **Temperature Exceedance**: Led by Docko, supported by Narayana.
- **Precipitation Exceedance (Short-run rainfall)**: Led by Koomi, supported by Shruti.
- **Onset/Cessation**: Led by Aryan, supported by Panchali.

> #### Exercise 1: Run Your First Forecast (15 min)
> 
> 1. Select the **AIFS v2 single** model.
> 2. Choose a start date within the model's valid range.
> 3. Set lead time to 5 days.
> 4. Select your country and use case.
> 5. Run the forecast and explore the results.
> 
> **Questions:**
> - How long did the forecast take to run?
> - What variables are shown in the output maps?
> - How does the forecast compare to ERA5 observations in the event movie?
{: .exercise}

<details>
<summary><strong>Discussion Points</strong></summary>
<br>

> #### Solution
> 
> - AIFS v2 single typically runs in ~2 minutes for a 10-day forecast.
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
| **attrs** | `model`, `model_name`, `init_source`, `members`, `synthetic` |

### 3.2 Loading Outputs in Python

```python
import xarray as xr

# Load a forecast
ds = xr.open_dataset("outputs/aifs_single/20241007T00_240h_m1.nc")

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

The visualization pipeline reads Demo 3 output files and dynamically fetches ERA5 observations from Google's ARCO dataset (cached locally in `outputs/_obs_cache/` if the cluster's primary ERA5 volume is unavailable). The selected use case dictates the movie variable:

- **Heat**: daily maximum 2m temperature
- **Precipitation**: daily rainfall accumulation

Movies are rendered on-demand and cached in `outputs/_movies/`.

> #### Synthetic Testing
> 
> To validate the pipeline without consuming GPU resources, enable **Use synthetic output** in the Streamlit sidebar and execute a run. This generates a mock NetCDF file that strictly adheres to the production data contract.
{: .tip}

## 4. Use-Case Customization

### 4.1 Temperature Exceedance

Focus: Heatwave tracking and daily maximum 2m temperature thresholds.

**Key parameters:**
- Variable: `t2m` (2-meter temperature)
- Metric: Daily maximum temperature
- Threshold: Region-specific (e.g., 35°C for heatwave conditions)

### 4.2 Precipitation Exceedance

Focus: Short-run rainfall accumulation and flood risk mapping.

**Key parameters:**
- Variable: `tp` (total precipitation)
- Metric: Accumulated rainfall over 24–72 hours
- Threshold: Region-specific (e.g., 50 mm/day for heavy rainfall)

### 4.3 Onset/Cessation

Focus: Seasonal transition markers and agricultural climate indices.

**Key parameters:**
- Variable: `tp` (precipitation)
- Metric: Wet-spell and dry-spell detection (see [Demo 4](04-weather-model.md) for full ROMP details)
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

2. **Provision environment**: Ensure the model's Python environment exists at `/opt/envs/<env>/bin/python` inside the container (see [Demo 1](01-setup.md) for environment details).

3. **Register model**: Set `status="ready"` for that model in `demo3/catalog.py`.

4. **Benchmark**: Add the measured runtime to `timings.json`.

> #### Data Contract Compliance
> 
> New models must strictly adhere to the output format (tp in mm, t2m in K, z500 in m² s⁻²). Non-compliant outputs will break the visualization pipelines.
{: .warning}

### 5.2 Core Directory Layout

*(Paths are relative to `ai_weather/demos/demo3/`)*

```text
app.py                 # Main Streamlit application: orchestration and results display
catalog.py             # Model registry (type, environment, lead-time limits, member counts)
regions.py             # Geographic mapping: program countries to padded plotting bounding boxes
contract.py            # Strict data contract: output-file format enforced for all runners
store.py               # Artifact management: storage paths and saved-run deduplication
jobs.py                # Asynchronous job launcher and progress tracker
movie.py               # Background event movie generator for loaded runs
results.py             # Pipeline visualization: model requirements, forecast maps, NetCDF export
event_movie/           # forecast_event_movie.py + Spark adaptation layer
theme.py               # UI styling, aligned with the Demo 5 platform design system
runners/run_model.py   # Universal runner entry point (+ synthetic output generator for testing)
runners/fgn_convert.py # FGN-specific starting condition converter from ECMWF open data
timings.json           # Empirically measured Spark runtimes displayed in the UI
docker/                # Containerization: Dockerfile, run.sh, per-environment dependency locks
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
> - **Speed vs. Accuracy**: Faster models (AIFS, FGN) sacrifice some accuracy for rapid inference.
> - **Resolution vs. Memory**: Higher resolution (0.25°) requires more GPU memory.
> - **Deterministic vs. Probabilistic**: Ensemble models provide uncertainty estimates but take longer.
{: .tip}

### 6.2 Initialization Overhead

First-time executions include weight downloads (see [Demo 1](01-setup.md) for download times per model). Subsequent runs start in seconds.

### 6.3 FGN (WeatherNext 2) Notes

FGN is Google DeepMind's Functional Generative Network for probabilistic forecasting. It requires special handling due to its unique data requirements. Detailed configuration is in [Demo 1, Section 6](01-setup.md).

- **Model variant**: `WeatherNextCyclones_Mini` (1° resolution, the only Mini weights published)
- **Starting conditions**: Google's sample file for 2024-10-07 (up to 7.5 days); converter for all other dates
- **Performance**: 10-day, 3-member forecast in ~2 minutes, 1.8 GB peak memory
- **Input cache**: 53 MB per start date, reused for subsequent runs

## Summary

- Demo 3 provides a unified interface for seven AI weather models.
- Containerized deployment ensures reproducibility across DGX Spark nodes.
- The 4-command workflow (build, run, get output, generate figure) simplifies execution.
- Use-case flags tailor outputs for temperature, precipitation, and onset forecasting.
- All models produce standardized NetCDF outputs for consistent analysis.
- Performance varies from 2 minutes (AIFS) to 26 minutes (Aurora) for 10-day forecasts.
- All configuration options are documented in [Demo 1](01-setup.md).

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
> ds = xr.open_dataset("outputs/graphcast/20241007T00_120h_m1.nc")
> ```
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
> print(ds.dims)   # {'ensemble': 1, 'lead_time': 21, 'lat': 721, 'lon': 1440}
> print(ds.coords)
> 
> # 2. Geopotential height at 48h
> z500_48h = ds["z500"].sel(lead_time=48)
> z500_height = z500_48h / 9.80665  # Convert to meters
> 
> # 3. Precipitation accumulation (24h to 48h)
> # tp is accumulated over 6h windows, so sum the relevant time steps
> tp_24_48 = ds["tp"].sel(lead_time=slice(24, 48)).sum(dim="lead_time")
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
> For comprehensive troubleshooting including GPU access, container conflicts, network issues, and model-specific problems, see [Demo 1: Troubleshooting Section](01-setup.md).
{: .tip}

| Symptom | Quick Resolution |
| --- | --- |
| Container won't start: "unresolvable CDI devices" | `systemctl --user restart docker && docker start demo3` |
| Port 8501 already in use | Use `ssh -L 8502:localhost:8501` and navigate to `:8502` |
| Model run fails during download | Retry or use `DEMO3_IFS_SOURCE=azure` |
| Out of memory errors | Check GPU memory with `nvidia-smi`; consider smaller lead times or fewer members |

## Next Steps

After completing this demo, proceed to **Demo 4** to learn systematic benchmarking of AI weather models against local observations for rainy season onset detection.
