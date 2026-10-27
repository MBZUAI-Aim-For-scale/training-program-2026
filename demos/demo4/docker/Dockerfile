# Demo 4 on a DGX Spark: docker/Dockerfile without the parts that come from
# demos/demo3/docker/Dockerfile (the model environments under /opt/envs, their
# locks and the GraphCast patch). What is left is the Streamlit site and the
# benchmarking code in the uv .venv.
#
# Benchmarking needs no GPU, so this starts from plain Python 3.12.3 (the version
# the NVIDIA PyTorch image had) instead of the NVIDIA image.
#
# The site, settings and benchmarking code come from this folder (demos/demo4/ui, apps and
# ai_weather); run_momp.py still comes from the repo root.
# Build (from the repo root, which the build needs for run_momp.py):
#   docker build -t demo4 -f demos/demo4/docker/Dockerfile .
# Run: see demos/demo4/docker/run.sh
FROM python:3.12.3

# Use bash for every RUN so `source` works (the default /bin/sh does not have it)
SHELL ["/bin/bash", "-c"]

# Fail fast: check that the requirements resolve before copying the code
RUN pip install uv
COPY demos/demo4/docker/requirements.txt /tmp/requirements.txt
# (to stdout: uv can't write its output file onto /dev/null inside a build)
RUN uv pip compile /tmp/requirements.txt --python-version 3.12 --quiet > /dev/null

WORKDIR /app

# uv environment for the app: create .venv inside /app
RUN uv venv .venv

# "Activate" .venv for every later build step, the CMD and docker exec sessions.
# PYTHONPATH=/app lets `import config` (and other top-level modules in /app) resolve
# when Streamlit runs ui/app.py.
ENV VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app:/app/ai_weather

# Same effect as the ENV above; kept so the activation is explicit
RUN source .venv/bin/activate \
 && uv pip install -r /tmp/requirements.txt

# Map outlines are fetched now so the page works without internet at the venue
RUN python -c "\
import cartopy.feature as cf, cartopy.io.shapereader as sr; \
[sr.natural_earth(resolution=r, category=c, name=n) for r in ('10m', '50m', '110m') \
 for c, n in [('physical', 'coastline'), ('physical', 'land'), ('physical', 'ocean'), \
              ('cultural', 'admin_0_boundary_lines_land'), ('cultural', 'admin_0_countries')]]"

# The code, after the packages so a code change doesn't reinstall them
COPY demos/demo4/ui /app/ui
COPY demos/demo4/apps /app/apps
COPY demos/demo4/ai_weather /app/ai_weather
COPY run_momp.py /app/

# ui/app.py writes this file on start when it's missing, and Streamlit then shows a
# "File change / Rerun" banner; writing it now leaves the app nothing to change. The [client]
# settings turn off Streamlit's own page list (built from ui/pages/) and Deploy button, which
# otherwise flash up until the site's CSS hides them.
RUN mkdir -p .streamlit \
 && printf '[theme]\nbase = "light"\n\n[client]\nshowSidebarNavigation = false\ntoolbarMode = "minimal"\n' \
      > .streamlit/config.toml

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.headless", "true", "--server.port", "8501", \
     "--browser.gatherUsageStats", "false"]
