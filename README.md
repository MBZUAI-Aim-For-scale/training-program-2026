# AI Forecasting Program

<a target="_blank" href="https://ai4ag.ai">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A hands-on training program for meteorologists and agricultural specialists from national
government services: generate, evaluate and deploy regionally tailored AI weather forecasts,
and pilot them in forecast-to-farmer advisory services.

**Lessons:** https://mbzuai-aim-for-scale.github.io/training-program-2026
(new to this? start with the [Beginner's Guide](demos/BEGINNER_GUIDE.md))

## Quick start (on a DGX Spark)

```bash
spark-setup dashboard     # the dashboard: http://localhost:8500
```

`spark-setup` is `python3 demos/demo1/spark_setup.py`, from any folder. Every run installs it, so the
first time (if it isn't there yet), use the long form.

From the dashboard, **Run checks** checks the Spark, and **Launch** builds a demo (only if its code changed) and starts it.
The country-data button downloads this Spark's data to `~/<country>_data`. Only one
demo runs at a time.

From your own computer, forward a port over SSH and open `http://localhost:<port>`:

```bash
ssh -N -L 8500:127.0.0.1:8500 <you>@<country>-spark
```

| Demo | What | Port | Start it alone |
|------|------|------|----------------|
| [1](demos/demo1/README.md) | Spark setup + dashboard | 8500 | `spark-setup dashboard` |
| 2 | Climate Data Tools (CDT) | 8504 | `bash demos/demo2/docker/run.sh` |
| [3](demos/demo3/README.md) | Run AI Forecast | 8502 | `bash demos/demo3/docker/run.sh` |
| [4](docs/_episodes/05-AI-Forest-test.md) | AfriClimate AI ForecastCheck (code in `demos/demo5`) | 8505 | `fcstcheck setup <country>` |
| [5](demos/demo4/README.md) | Onset Benchmarking (code in `demos/demo4`) | 8503 | `bash demos/demo4/docker/run.sh start` |
| AI Almanac | Assess, blend and forecast with AI weather models | 8765 | the dashboard's **AI Almanac** tab (`~/ai-almanac`) |

Other setup commands: `spark-setup --help`. To stop everything:
`bash demos/demo1/shutdown.sh`.

**Data:** the demos read this Spark's country data first, and the program's bucket
(`gs://aim4scale_training_2026`) for anything else. Settings go in `.env` (see `.env.example`).

## Project organization

```
├── demos/
│   ├── BEGINNER_GUIDE.md   step-by-step guide for every demo
│   ├── intro-to-spark/     Intro to DGX Spark: background docs, Africa-focused labs 00-13 in five topic folders, docker/
│   ├── demo1/              Spark setup: dashboard, spark_setup.py, demos.toml (the demo list), country_data.py
│   ├── demo2/              CDT (Climate Data Tools) container
│   ├── demo3/              Demo 3, Run AI Forecast (Streamlit app + runners)
│   ├── demo4/              Demo 5, Onset Benchmarking: ai_weather package, apps/, ui/, docker/, tests/
│   └── demo5/              Demo 4, AfriClimate AI ForecastCheck launcher (run.sh; its code is cloned into demo5/forecastcheck/)
├── docs/                   lesson site (GitHub Pages); lessons in docs/_episodes/
├── apps/onset_blending/    onset blending pipeline (run_blending.py)
├── data/ROMP_OUTPUT/       sample benchmark results
├── tests/                  repository-level tests (demos/demo4 layout, country benchmarks)
├── run_momp.py             Demo 5 (onset) benchmark CLI
├── run_app.py              Demo 5 (onset) UI without Docker (streamlit)
├── run.sh, docker-compose.yml, config.py
│                           shims that forward to Demo 5 (demos/demo4)
├── pyproject.toml, setup.py, uv.lock, requirements.txt
│                           the ai_weather package (Demo 5's code)
└── models/, references/, reports/   placeholders
```
