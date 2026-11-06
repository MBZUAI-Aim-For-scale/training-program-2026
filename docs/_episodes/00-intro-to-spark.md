---
title: "Intro to DGX Spark"
teaching: 15
exercises: 120
questions:
- "What is a DGX Spark, and how is it different from a GPU server or the cloud?"
- "Which weather, climate and agriculture workloads fit on it?"
- "How do AI forecasts become decisions: farm advisories, crop stress, cyclone tracks, solar energy?"
- "How do we know whether a forecast is any good, and which data does a forecast service depend on?"
objectives:
- "Know the Spark's specs, unified memory, and how to share one Spark between teams."
- "Run AI forecasts (SFNO, AIFS-ENS, Aurora) and an ensemble, in JupyterLab on the Spark."
- "Verify forecasts against ERA5, baselines and real weather stations."
- "Turn forecasts into applications: SMS advisories from a local LLM, crop and livestock stress, solar output."
- "Work with climate data: rainfall normals and onset, station bias correction, the data behind a forecast."
keypoints:
- "Read the six short background pages first (15 minutes)."
- "Lab numbers are the same everywhere; all labs share one outputs folder, so later labs reuse earlier results (Lab 04's ensemble feeds Labs 05, 06, 08, 10 and 12)."
- "Lab 00 starts JupyterLab with the labs on the Spark, on a port of your own."
- "Model weights (~35 GB) download on first use and are cached on the Spark."
- "Thresholds in the labs are placeholders for country teams to set."
---

# Intro to DGX Spark

What a DGX Spark is, what it can do for weather, climate and agriculture, and fourteen hands-on labs that take
AI forecasts all the way to farm advisories, crop stress, cyclone tracks, solar energy and verification against
real stations, plus the data a forecast service depends on. It comes before [Demo 1](/training-program-2026/01-setup/) (setting up
the lab) and complements [Demo 3](/training-program-2026/03-forecast/) (the forecast lab).

Everything is in the repository's [`demos/intro-to-spark`](https://github.com/MBZUAI-Aim-For-scale/training-program-2026/tree/main/demos/intro-to-spark)
folder: its [README](https://github.com/MBZUAI-Aim-For-scale/training-program-2026/blob/main/demos/intro-to-spark/README.md)
has every lab with what it uses and how long it takes.

## Read first (15 min)

| | |
|---|---|
| [What is DGX Spark?](https://github.com/MBZUAI-Aim-For-scale/training-program-2026/blob/main/demos/intro-to-spark/docs/01-what-is-dgx-spark.md) | Specs and the three things that differ from a GPU server |
| [Capabilities](https://github.com/MBZUAI-Aim-For-scale/training-program-2026/blob/main/demos/intro-to-spark/docs/02-capabilities.md) | Which weather and climate workloads fit |
| [Best practices](https://github.com/MBZUAI-Aim-For-scale/training-program-2026/blob/main/demos/intro-to-spark/docs/03-best-practices.md) | Environment, unified memory, sharing one Spark between teams |
| [Use cases](https://github.com/MBZUAI-Aim-For-scale/training-program-2026/blob/main/demos/intro-to-spark/docs/04-use-cases-weather-climate-ag.md) | Decisions in agriculture, disasters, energy and weather services, mapped to labs |
| [Spark vs cloud one-pager](https://github.com/MBZUAI-Aim-For-scale/training-program-2026/blob/main/demos/intro-to-spark/docs/05-hardware-one-pager-spark-vs-cloud.md) | Hardware considerations, with numbers measured on a Spark |
| [Data one-pager](https://github.com/MBZUAI-Aim-For-scale/training-program-2026/blob/main/demos/intro-to-spark/docs/06-data-guideline-one-pager.md) | The data side of running AI models: inputs, timeliness, storage, observations |

## The labs

| Topic | Labs |
|---|---|
| 1. Getting started | 00 connect to the Spark · 01 know your Spark |
| 2. Forecasting | 02 first AI forecast · 04 rainfall ensemble · 07 Cyclone Idai hindcast |
| 3. Verification | 03 verify a forecast · 09 model scorecard · 13 verification methods |
| 4. Applications | 05 farm advisories from a local LLM · 06 crop and livestock heat and water stress · 08 solar for mini-grids and pumps |
| 5. Data and climate | 10 rainfall normals and onset · 11 station bias correction · 12 the data behind AI weather |

## Starting the labs

[Lab 00](https://github.com/MBZUAI-Aim-For-scale/training-program-2026/blob/main/demos/intro-to-spark/labs/1-getting-started/lab00_connect_to_spark.md)
builds the labs' image and starts JupyterLab on the Spark, on a port of your own (one per person or team).
The models download on first use.
