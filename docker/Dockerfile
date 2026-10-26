# Demo 3 and 4 on a DGX Spark: Streamlit UI and model stacks in one image.
FROM nvcr.io/nvidia/pytorch:25.10-py3

SHELL ["/bin/bash", "-c"]

# Environment configuration
ENV PIP_CONSTRAINT= \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    AI_WEATHER_ENV_ROOT=/opt/envs \
    DEMO3_ENV_ROOT=/opt/envs \
    VIRTUAL_ENV=/opt/envs/app \
    PATH=/opt/envs/app/bin:/opt/envs/ui/bin:$PATH \
    PYTHONPATH=/app:/app/ai_weather

# Install uv package manager
RUN pip install uv
<<<<<<< HEAD

# Install gcsfuse and FUSE utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg lsb-release fuse \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/gcsfuse-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/gcsfuse-archive-keyring.gpg] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gcsfuse.list \
    && apt-get update \
    && apt-get install -y gcsfuse \
    && rm -rf /var/lib/apt/lists/*

# Pre-create data and output directories
RUN mkdir -p /app/outputs /app/apps/forecasting/outputs /app/data /root/.cache /etc/gcp

# Resolve UI requirements
COPY docker/requirements.txt /tmp/requirements.raw.txt
RUN grep -vE '^-e |file://|^appnope==' /tmp/requirements.raw.txt > /tmp/requirements.txt \
 && uv pip compile /tmp/requirements.txt --python-version 3.12 -q -o /tmp/requirements.resolved.txt

COPY docker/locks /tmp/locks

# 1. Atlas CRPS, Aurora 1.5, AIFS v2 single (Earth2Studio 0.18)
RUN python -m venv --system-site-packages /opt/envs/e2s018 \
 && NATTEN_CUDA_ARCH="12.1" NATTEN_N_WORKERS=16 uv pip install --python /opt/envs/e2s018 --no-deps --no-build-isolation -r /tmp/locks/e2s018.txt \
 && /opt/envs/e2s018/bin/python -c "from earth2studio.models.px import AtlasCRPS, Aurora1p5, AIFS2"

# 2. AIFS v2 ENS
RUN python -m venv --system-site-packages /opt/envs/e2s018ens \
 && uv pip install --python /opt/envs/e2s018ens --no-deps -r /tmp/locks/e2s018ens.txt \
 && /opt/envs/e2s018ens/bin/python -c "from earth2studio.models.px import AIFS2ENS"

# 3. GraphCast and NeuralGCM (JAX)
RUN python -m venv --system-site-packages /opt/envs/graphcast \
 && uv pip install --python /opt/envs/graphcast --no-deps -r /tmp/locks/graphcast.txt \
 && /opt/envs/graphcast/bin/python -c "import graphcast, neuralgcm, jax; from earth2studio.models.px import GraphCastOperational" \
 && ln -s /opt/envs/graphcast /opt/envs/neuralgcm

# 4. FGN (WeatherNext 2, JAX)
RUN python -m venv --system-site-packages /opt/envs/fgn \
 && uv pip install --python /opt/envs/fgn --no-deps -r /tmp/locks/fgn.txt \
 && /opt/envs/fgn/bin/python -c "from weathernext.weathernext2 import fgn; import xarray_jax"

# 5. Streamlit UI environment
RUN python -m venv /opt/envs/ui \
 && uv pip install --python /opt/envs/ui -r /tmp/requirements.txt \
 && /opt/envs/ui/bin/python -c "\
import cartopy.feature as cf, cartopy.io.shapereader as sr; \
[sr.natural_earth(resolution=r, category=c, name=n) for r in ('10m', '50m', '110m') \
 for c, n in [('physical', 'coastline'), ('physical', 'land'), ('physical', 'ocean'), \
              ('cultural', 'admin_0_boundary_lines_land'), ('cultural', 'admin_0_countries')]]"

# Fix Earth2Studio 0.16 GraphCast xarray dataset compatibility
RUN cd /opt/envs/graphcast/lib/python3.12/site-packages/earth2studio/models/px \
 && sed -i -E 's/^( +)(inputs|targets_template|forcings) = xr\.Dataset\(\2\)$/\1\2 = \2.copy()/' \
      graphcast_operational.py graphcast_small.py \
 && grep -q "forcings = forcings.copy()" graphcast_operational.py \
 && grep -q "forcings = forcings.copy()" graphcast_small.py

WORKDIR /app

# Create isolated application virtual environment in /opt/envs/app
RUN uv venv /opt/envs/app \
 && uv pip install --python /opt/envs/app -r /tmp/requirements.resolved.txt

COPY . /app

RUN if [ -f pyproject.toml ] || [ -f setup.py ]; then uv pip install --python /opt/envs/app --no-deps -e .; fi \
 && /opt/envs/app/bin/python -c "import config; print('config OK:', config.__file__)"

EXPOSE 8501 8888

CMD ["streamlit", "run", "ui/app.py", "--server.headless", "true", "--server.port", "8501", \
     "--browser.gatherUsageStats", "false"]
