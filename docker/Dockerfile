FROM nvcr.io/nvidia/pytorch:25.10-py3

# Use bash for every RUN step
SHELL ["/bin/bash", "-c"]

# Environment defaults for building & serving
ENV PIP_CONSTRAINT= \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    AI_WEATHER_ENV_ROOT=/opt/envs

RUN pip install uv

# Pre-create data/output directories so the server starts cleanly even without host mounts
RUN mkdir -p /app/outputs /app/apps/forecasting/outputs /app/data /root/.cache

# Resolve UI requirements
COPY docker/requirements.txt /tmp/requirements.raw.txt
RUN grep -vE '^-e |file://|^appnope==' /tmp/requirements.raw.txt > /tmp/requirements.txt \
 && uv pip compile /tmp/requirements.txt --python-version 3.12 -q -o /tmp/requirements.resolved.txt

COPY docker/locks /tmp/locks

# 1. Atlas CRPS, Aurora 1.5, AIFS v2 single (Earth2Studio 0.18)
# Pre-install build tools (ninja, cmake, wheel) inside venv so NATTEN compiles without isolation
RUN python -m venv --system-site-packages /opt/envs/e2s018 \
 && /opt/envs/e2s018/bin/pip install ninja cmake wheel \
 && NATTEN_CUDA_ARCH="8.0;8.6;8.9;9.0;12.0;12.1" NATTEN_N_WORKERS=16 \
    /opt/envs/e2s018/bin/pip install --no-deps --no-build-isolation -r /tmp/locks/e2s018.txt \
 && /opt/envs/e2s018/bin/python -c "from earth2studio.models.px import AtlasCRPS, Aurora1p5, AIFS2"

# 2. AIFS v2 ENS
RUN python -m venv --system-site-packages /opt/envs/e2s018ens \
 && /opt/envs/e2s018ens/bin/pip install --no-deps -r /tmp/locks/e2s018ens.txt \
 && /opt/envs/e2s018ens/bin/python -c "from earth2studio.models.px import AIFS2ENS"

# 3. GraphCast and NeuralGCM (JAX)
RUN python -m venv --system-site-packages /opt/envs/graphcast \
 && /opt/envs/graphcast/bin/pip install --no-deps -r /tmp/locks/graphcast.txt \
 && /opt/envs/graphcast/bin/python -c "import graphcast, neuralgcm, jax; from earth2studio.models.px import GraphCastOperational" \
 && ln -s /opt/envs/graphcast /opt/envs/neuralgcm

# 4. FGN (WeatherNext 2, JAX)
RUN python -m venv --system-site-packages /opt/envs/fgn \
 && /opt/envs/fgn/bin/pip install --no-deps -r /tmp/locks/fgn.txt \
 && /opt/envs/fgn/bin/python -c "from weathernext.weathernext2 import fgn; import xarray_jax"

# 5. Streamlit UI environment
RUN python -m venv /opt/envs/ui \
 && /opt/envs/ui/bin/pip install -r /tmp/requirements.txt \
 && /opt/envs/ui/bin/python -c "\
import cartopy.feature as cf, cartopy.io.shapereader as sr; \
[sr.natural_earth(resolution=r, category=c, name=n) for r in ('10m', '50m', '110m') \
 for c, n in [('physical', 'coastline'), ('physical', 'land'), ('physical', 'ocean'), \
              ('cultural', 'admin_0_boundary_lines_land'), ('cultural', 'admin_0_countries')]]"

# 6. Fix Earth2Studio 0.16 GraphCast xarray dataset compatibility
RUN cd /opt/envs/graphcast/lib/python3.12/site-packages/earth2studio/models/px \
 && sed -i -E 's/^( +)(inputs|targets_template|forcings) = xr\.Dataset\(\2\)$/\1\2 = \2.copy()/' \
      graphcast_operational.py graphcast_small.py \
 && grep -q "forcings = forcings.copy()" graphcast_operational.py \
 && grep -q "forcings = forcings.copy()" graphcast_small.py

WORKDIR /app
COPY . /app

ENV PATH=/opt/envs/ui/bin:$PATH \
      DEMO3_ENV_ROOT=/opt/envs

ENV VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:/opt/envs/ui/bin:$PATH \
    PYTHONPATH=/app:/app/ai_weather \
    AI_WEATHER_ENV_ROOT=/opt/envs

# Application environment setup
RUN rm -rf .venv && uv venv .venv


RUN source .venv/bin/activate \
 && uv pip install -r /tmp/requirements.txt \
 && if [ -f pyproject.toml ] || [ -f setup.py ]; then uv pip install --no-deps -e .; fi \
 && python -c "import config; print('config OK:', config.__file__)"

EXPOSE 8501

# Serve Streamlit listening on 0.0.0.0
CMD ["streamlit", "run", "ui/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
