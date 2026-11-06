# Intro to DGX Spark: AI weather for Africa

What a DGX Spark is, what it can do for weather, climate and agriculture, and fourteen hands-on labs that take
AI forecasts all the way to farm advisories, crop stress, cyclone tracks, solar energy and verification
against real stations, plus the data a forecast service depends on. Every lab is about Africa. Thresholds in the labs are placeholders for country teams to set.

Complements [Demo 1](../demo1/) (machine setup) and [Demo 3](../demo3/) (the forecast lab).

## Read first (15 min)

| | |
|---|---|
| [What is DGX Spark?](docs/01-what-is-dgx-spark.md) | Specs and the three things that differ from a GPU server |
| [Capabilities](docs/02-capabilities.md) | Which weather and climate workloads fit |
| [Best practices](docs/03-best-practices.md) | Environment, unified memory, sharing one Spark between teams |
| [Use cases for Africa](docs/04-use-cases-weather-climate-ag.md) | Decisions in agriculture, disasters, energy and weather services, mapped to labs |
| [Spark vs cloud one-pager](docs/05-hardware-one-pager-spark-vs-cloud.md) | Hardware considerations, with numbers measured on a Spark |
| [Data one-pager](docs/06-data-guideline-one-pager.md) | The data side of running AI models: inputs, timeliness, storage, observations |

## Labs

Grouped by topic. Lab numbers are the same everywhere, so "Lab 04" always means the same notebook. All labs
write to one shared `labs/outputs/` folder, which is how later labs reuse earlier results (Lab 04's ensemble is
read by Labs 05, 06, 08, 10 and 12).

### 1. Getting started — [`labs/1-getting-started/`](labs/1-getting-started/)

| Lab | What you do | Earth-2 pieces | Internet | Time on Spark |
|---|---|---|---|---|
| [00](labs/1-getting-started/lab00_connect_to_spark.md) | SSH, lab container, Jupyter via tunnel | — | Build only | 15 min |
| [01](labs/1-getting-started/lab01_know_your_spark.ipynb) | Know your Spark: unified memory, GPU speed vs memory bandwidth, sizing a forecast, sharing the box | — | No | <1 min |

### 2. Forecasting — [`labs/2-forecasting/`](labs/2-forecasting/)

| Lab | What you do | Earth-2 pieces | Internet | Time on Spark |
|---|---|---|---|---|
| [02](labs/2-forecasting/lab02_first_forecast_africa.ipynb) | First AI forecast for Africa: heat, wind, moisture maps and a city table | SFNO, GFS | Yes | 1.5 min |
| [04](labs/2-forecasting/lab04_africa_rainfall_ensemble.ipynb) | 8-member rainfall ensemble for all of Africa: totals and wet-spell probability. **Saves the data Labs 05, 06, 08, 10 use** | AIFS-ENS, IFS | Yes | 12 min |
| [07](labs/2-forecasting/lab07_cyclone_idai_hindcast.ipynb) | Cyclone Idai (2019): rerun three AI models 5 days before landfall, compare with the real track | SFNO, AIFS, Aurora, IBTrACS | Yes | 9 min |

### 3. Verification — [`labs/3-verification/`](labs/3-verification/)

| Lab | What you do | Earth-2 pieces | Internet | Time on Spark |
|---|---|---|---|---|
| [03](labs/3-verification/lab03_verify_east_africa.ipynb) | Is it any good? Score vs ERA5 and a baseline, then vs real weather stations | SFNO, ARCO ERA5, GHCN-Daily | Yes | 2 min |
| [09](labs/3-verification/lab09_africa_model_scorecard.ipynb) | Which model should Africa trust? 3 models x 4 dates scored over Africa | SFNO, AIFS, Aurora, ERA5 | Yes | 26 min |
| [13](labs/3-verification/lab13_verification_methods.ipynb) | Verification methods on the April 2024 East Africa floods: bias and RMSE, rain hit/miss scores, Fractions Skill Score, ensemble reliability (CRPS, Brier, ROC), rain gauges | AIFS-ENS, ERA5, `earth2studio.statistics`, GHCN-Daily | Yes | 10 min first run, 1 min after |

### 4. Applications — [`labs/4-applications/`](labs/4-applications/)

| Lab | What you do | Earth-2 pieces | Internet | Time on Spark |
|---|---|---|---|---|
| [05](labs/4-applications/lab05_llm_farm_advisories.ipynb) | Farmer SMS advisories written by a local LLM, with automatic fact checks | Qwen3-8B on the Spark | First run (16 GB) | 3 min |
| [06](labs/4-applications/lab06_crop_heat_water_stress.ipynb) | Crop and livestock stress: hot days, vapour-pressure deficit, dry spells, cattle heat index | AIFS-ENS output | No | <1 min |
| [08](labs/4-applications/lab08_solar_minigrids.ipynb) | Solar for mini-grids and irrigation pumps: kWh per kWp and m³ of water per day | AIFS-ENS, SFNO + SolarRadiationAFNO | Yes | 2 min |

### 5. Data and climate — [`labs/5-data-and-climate/`](labs/5-data-and-climate/)

| Lab | What you do | Earth-2 pieces | Internet | Time on Spark |
|---|---|---|---|---|
| [10](labs/5-data-and-climate/lab10_rainfall_climate_onset.ipynb) | 42 years of rainfall: normals, onset of the rains and its trend (on the GPU), this week vs normal | ERA5 (WeatherBench 2), CuPy | First run (~6 min) | <1 min cached |
| [11](labs/5-data-and-climate/lab11_station_bias_correction.ipynb) | Make gridded data fit your stations: per-station and ML corrections | ERA5, GHCN-Daily, scikit-learn | First run | 3.5 min |
| [12](labs/5-data-and-climate/lab12_data_for_ai_weather.ipynb) | The data behind a forecast: model inputs vs data sources, data freshness, download and storage size, unit pitfalls, station coverage | SFNO, AIFS-ENS, Aurora inputs; GFS, IFS, ERA5, GHCN-Daily | Yes | 1.5 min |

## Quick start (on the Spark)

```bash
git clone https://github.com/MBZUAI-Aim-For-scale/training-program-2026.git
cd training-program-2026
docker build -t spark-intro demos/intro-to-spark/docker      # once per Spark, 30-60 min
bash demos/intro-to-spark/docker/run.sh                       # prints the Jupyter token and tunnel command
```

Details and troubleshooting: [Lab 00](labs/1-getting-started/lab00_connect_to_spark.md).

## Running this with several teams on one Spark

Labs 04, 07, 09 and 13 each use 60–66 GiB of the Spark's 128 GB and run for 9–26 minutes. Two of them at once can
exhaust memory, and Docker's `--memory` limit does not cap GPU memory on Spark (Lab 01). Plan for it:

- **Pre-run Labs 04, 07, 09 and 13** on each Spark before the session. For Lab 04, copy the instructor's
  `outputs/lab04_africa_ensemble.zarr` (375 MB) into each team's `labs/outputs/` and set `USE_CACHED = True`:
  the lab then needs no GPU or internet, and still works on a day the IFS data is unavailable.
- With that run in place, all teams can do Labs 04, 06 and 10 at once (post-processing only). Labs 05 (LLM,
  ~38 GiB) and 08 (SFNO + solar model, ~33 GiB) still load models: one team at a time, or an instructor demo.
- Let one team at a time run a heavy lab live, as a demonstration.

## Tested

All labs ran on one DGX Spark (GB10, driver 580.95, NGC PyTorch 26.06, Earth2Studio commit `d654ad2`),
2026-10-29/30, in an image built with the same steps as `docker/Dockerfile`. Figures from that run are in
[`results/`](results/). Times include data downloads, with model weights already cached.

| Lab | Wall | Peak memory (GiB, system) | Result |
|---|---|---|---|
| 01 | 12 s | 16 | 36.5 TFLOPS bf16, 18.6 tf32, 7.1 fp32; 230 GB/s copy |
| 02 | 1.5 min | 35 | Forecast in 42 s including the GFS download |
| 03 | 1.9 min | 35 | SFNO beats "yesterday again" at 16 of 20 lead times; daily max vs 13 stations: SFNO and ERA5 both 1.9 °C mean error |
| 04 | 11.8 min | 63 | Ensemble 10.8 min, 36 GiB GPU |
| 05 | 2.8 min | 38 | 7 tokens/s for one message, 20 batched; 4 of 5 advisories pass the fact check, Kampala flagged for review ([output](results/lab05_advisories.md)) |
| 06 | 15 s | 7 | Post-processing of Lab 04 |
| 07 | 8.6 min | 64 | All three models put landfall 160–180 km south of Beira; central pressure 970–990 hPa vs 940–960 observed |
| 08 | 1.9 min | 33 | The two solar forecasts agree to ±0.7 kWh/m²/day at 90% of points |
| 09 | 25.7 min | 66 | AIFS has the lowest day-5 error on t2m, msl, 10 m wind and z500 |
| 10 | 6.2 min first run, 0.4 min cached | 8 | Onset of the rains: CPU 445 ms, GPU 37 ms |
| 11 | 3.3 min | 7 | Daily max error: raw 2.16 °C, per-station correction 0.82, one ML model on unseen stations 1.86 |
| 12 (2026-11-02) | 1.5 min | 21 | SFNO inputs from GFS: 73 fields, 62 MB in 11 s; GFS lacks 4 AIFS-ENS inputs; matching Zarr chunks 10–30x faster; 10 public stations per M km² in Africa vs 240 in Europe |
| 13 (2026-11-03) | 10 min first run, 1.0 min cached | 23 (cached) | Day-5 t2m RMSE: ensemble mean 1.40 °C, one member 1.81, persistence 2.35, climatology 2.74; rain > 10 mm/day: CSI 0.37, Brier skill 0.39, ROC area 0.93; rain ensemble over-confident (spread/error 0.5–0.8) |

Known after the run: Lab 03 saved a blank figure (the plot was finished in a later cell). That is fixed here but
has not been rerun; its figure is not in `results/`.

## Editing the labs

Each lab's source is the `.py` file (cells marked with `# %%`), which diffs cleanly. After editing, regenerate the
notebooks: `python3 demos/intro-to-spark/scripts/py2nb.py demos/intro-to-spark/labs/*/lab*.py`.
