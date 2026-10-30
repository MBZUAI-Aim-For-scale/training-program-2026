# AI Forecasting Program

<a target="_blank" href="https://ai4ag.ai">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A hands-on training program bringing together meteorologists and agricultural specialists from national government services to build capacity in AI-powered weather forecasting. Participants learn to generate, evaluate, and deploy regionally tailored AI forecasts, translating cutting-edge technology into practical agricultural advisory services that can reach smallholder farmers at scale.

The program is designed to provide an interactive learning experience over the AI weather training program, so we can move from concepts to pilot evaluation as part of an international community of practice. The 2026 program goals aim to help participants from weather and agricultural government institutions think through together how they could pilot an AI-based weather model in their operational workflows as part of forecast-to-farmer decision support. To achieve this, we have a track-based learning approach to engage participants (some sessions are joint across weather and agriculture, while others are separate to allow a deeper dive across the pipeline).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── demos              <- src for each demos
│   ├── intro-to-spark <- Intro to DGX Spark: background docs + Africa-focused labs 00-11
│   ├── demo3          <- Demo3 source code
│   ├── demo4          <- Demo4 source code
│   ├── demo5          <- Demo5 source code
│   └── demo6          <- Demo6 source code
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│   └── _episodes      <-  md lesson for each demos
│   └── _layouts       <-  layout of the lesson for the training on GitHub pages
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ai_weather and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│   └── slide          <-  quarto revealjs slide for ppt 
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ai_weather   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ai_weather a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

