# Intro to DGX Spark

What a DGX Spark is, what it can do for weather, climate and agriculture, how to use it well, and a first
AI forecast taken all the way to a farm advisory and a verification score. Seven short labs, about 1.5 hours.

Complements [Demo 1](../demo1/) (machine setup) and [Demo 3](../demo3/) (the full forecast lab):
these labs are the hands-on "what is this machine and how do I use it" step before them.

## Read first (15 min)

| | |
|---|---|
| [What is DGX Spark?](docs/01-what-is-dgx-spark.md) | Specs and the three things that differ from a GPU server |
| [Capabilities](docs/02-capabilities.md) | Which weather/climate workloads fit |
| [Best practices](docs/03-best-practices.md) | Environment, unified memory, sharing one Spark between teams |
| [Use cases](docs/04-use-cases-weather-climate-ag.md) | Weather, climate and agriculture decisions mapped to forecast variables |
| [Spark vs cloud one-pager](docs/05-hardware-one-pager-spark-vs-cloud.md) | Hardware considerations, with numbers measured on a Spark |

## Labs

| Lab | Topic | Internet | Time |
|---|---|---|---|
| [00](labs/lab00_connect_to_spark.md) | Connect: SSH, lab container, Jupyter through a tunnel | Build only | 15 min |
| [01](labs/lab01_system_tour.ipynb) | System tour: aarch64, unified memory, GPU kernel check | No | 10 min |
| [02](labs/lab02_gpu_benchmark.ipynb) | GPU throughput by precision, memory bandwidth | No | 10 min |
| [03](labs/lab03_first_forecast.ipynb) | First AI forecast (SFNO + GFS) and a UAE map | Yes | 15 min |
| [04](labs/lab04_memory_and_sizing.ipynb) | Memory budgeting, ensemble sizing, sharing the box | No | 10 min |
| [05](labs/lab05_forecast_to_farm_alert.ipynb) | Forecast -> thresholds -> farm advisory draft | No (uses Lab 03 output) | 15 min |
| [06](labs/lab06_verify_against_era5.ipynb) | Verify against ERA5, beat a baseline, threshold hit rates | Yes | 20 min |

## Quick start (on the Spark)

```bash
git clone https://github.com/MBZUAI-Aim-For-scale/training-program-2026.git
cd training-program-2026
docker build -t spark-intro demos/intro-to-spark/docker      # once, ~15 min
bash demos/intro-to-spark/docker/run.sh                       # prints the Jupyter token and tunnel command
```

Details and troubleshooting: [Lab 00](labs/lab00_connect_to_spark.md).

## Tested

Labs 01–06 ran end to end on a DGX Spark (GB10, driver 580.95, NGC PyTorch 26.06, Earth2Studio built from
commit `d654ad2`). Figures from that run are in [`results/`](results/). The run used a larger image than
`docker/Dockerfile`, which installs only what these labs need; that trimmed image has not been built yet.
The Lab 05 figure shows the earlier 42 °C threshold; the lab now uses 35 °C so the alert fires in November.

## Editing the labs

Each lab's source is the `.py` file (cells marked with `# %%`), which diffs cleanly. After editing, regenerate the
notebooks: `python3 demos/intro-to-spark/scripts/py2nb.py demos/intro-to-spark/labs/lab0*.py`.
