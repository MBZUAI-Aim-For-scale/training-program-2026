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
| [3](demos/demo3/README.md) | AI forecast lab | 8502 | `bash demos/demo3/docker/run.sh` |
| [4](demos/demo4/README.md) | Forecasting & benchmarking | 8503 | `bash demos/demo4/docker/run.sh start` |
| [5](docs/_episodes/05-AI-Forest-test.md) | ForecastCheck | 8505 | `fcstcheck setup <country>` |
| AI Almanac | Assess, blend and forecast with AI weather models | 8765 | the dashboard's **AI Almanac** tab (`~/ai-almanac`) |

Other setup commands: `spark-setup --help`. To stop everything:
`bash demos/demo1/shutdown.sh`.

**Data:** the demos read this Spark's country data first, and the program's bucket
(`gs://aim4scale_training_2026`) for anything else. Settings go in `.env` (see `.env.example`).

## Project organization

```
├── demos/
│   ├── BEGINNER_GUIDE.md   step-by-step guide for every demo
│   ├── demo1/              Spark setup: dashboard, spark_setup.py, demos.toml (the demo list), country_data.py
│   ├── demo2/              CDT (Climate Data Tools) container
│   ├── demo3/              AI forecast lab (Streamlit app + runners)
│   ├── demo4/              forecasting & benchmarking: ai_weather package, apps/, ui/, docker/, tests/
│   └── demo5/              ForecastCheck launcher (run.sh; its code is cloned into demo5/forecastcheck/)
├── docs/                   lesson site (GitHub Pages); lessons in docs/_episodes/
├── apps/onset_blending/    onset blending pipeline (run_blending.py)
├── data/ROMP_OUTPUT/       sample benchmark results
├── tests/                  repository-level tests (Demo 4 layout, country benchmarks)
├── run_momp.py             Demo 4 benchmark CLI
├── run_app.py              Demo 4 UI without Docker (streamlit)
├── run.sh, docker-compose.yml, config.py
│                           shims that forward to Demo 4 (demos/demo4)
├── pyproject.toml, setup.py, uv.lock, requirements.txt
│                           the ai_weather package (Demo 4's code)
└── models/, references/, reports/   placeholders
```
