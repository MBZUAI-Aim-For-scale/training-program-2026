# Demo 3 on a DGX Spark: the Streamlit page and every model in one image.
#
# Build (from the demos/demo3/ folder, on the Spark; aarch64 only):
#   docker build -t demo3 -f docker/Dockerfile .
# Run: see docker/run.sh
#
# Each model family needs conflicting libraries, so each gets its own virtual
# environment under /opt/envs/<env> (the names in demo3/catalog.py). They share
# the NGC container's PyTorch / CUDA builds through --system-site-packages, and
# docker/locks/<env>.txt pins every other package to the versions timed on the
# Spark. Model weights are not baked in: they download on first use into
# /root/.cache, which run.sh keeps on the host.
FROM nvcr.io/nvidia/pytorch:25.10-py3

# The locks are exact; NGC's pip constraint file would fight them.
ENV PIP_CONSTRAINT= \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY docker/locks /tmp/locks

# Atlas CRPS, Aurora 1.5, AIFS v2 single (Earth2Studio 0.18). No build isolation:
# NATTEN compiles from source and must build against the container's PyTorch.
# There is no GPU during a build, so name the Spark's GB10 (sm_121) explicitly.
RUN python -m venv --system-site-packages /opt/envs/e2s018 \
 && NATTEN_CUDA_ARCH="12.1" NATTEN_N_WORKERS=16 /opt/envs/e2s018/bin/pip install --no-deps --no-build-isolation -r /tmp/locks/e2s018.txt \
 && /opt/envs/e2s018/bin/python -c "from earth2studio.models.px import AtlasCRPS, Aurora1p5, AIFS2"

# AIFS v2 ENS (needs a different anemoi-models)
RUN python -m venv --system-site-packages /opt/envs/e2s018ens \
 && /opt/envs/e2s018ens/bin/pip install --no-deps -r /tmp/locks/e2s018ens.txt \
 && /opt/envs/e2s018ens/bin/python -c "from earth2studio.models.px import AIFS2ENS"

# GraphCast and NeuralGCM (JAX)
RUN python -m venv --system-site-packages /opt/envs/graphcast \
 && /opt/envs/graphcast/bin/pip install --no-deps -r /tmp/locks/graphcast.txt \
 && /opt/envs/graphcast/bin/python -c "import graphcast, neuralgcm, jax; from earth2studio.models.px import GraphCastOperational" \
 && ln -s /opt/envs/graphcast /opt/envs/neuralgcm

# FGN (WeatherNext 2, JAX)
RUN python -m venv --system-site-packages /opt/envs/fgn \
 && /opt/envs/fgn/bin/pip install --no-deps -r /tmp/locks/fgn.txt \
 && /opt/envs/fgn/bin/python -c "from weathernext.weathernext2 import fgn; import xarray_jax"

# The Streamlit page and Panchali's event movie, kept apart from the model stacks
COPY docker/requirements-ui.txt /tmp/requirements-ui.txt
RUN python -m venv /opt/envs/ui \
 && /opt/envs/ui/bin/pip install -r /tmp/requirements-ui.txt \
 && /opt/envs/ui/bin/python -c "\
import cartopy.feature as cf, cartopy.io.shapereader as sr; \
[sr.natural_earth(resolution=r, category=c, name=n) for r in ('10m', '50m', '110m') \
 for c, n in [('physical', 'coastline'), ('physical', 'land'), ('physical', 'ocean'), \
              ('cultural', 'admin_0_boundary_lines_land'), ('cultural', 'admin_0_countries')]]"
# (map outlines are fetched now so the page works without internet at the venue)

# Earth2Studio 0.16's GraphCast wraps Datasets in xr.Dataset(...), which newer
# xarray rejects; copy them instead (the same fix the summer environment carried).
RUN cd /opt/envs/graphcast/lib/python3.12/site-packages/earth2studio/models/px \
 && sed -i -E 's/^( +)(inputs|targets_template|forcings) = xr\.Dataset\(\2\)$/\1\2 = \2.copy()/' \
      graphcast_operational.py graphcast_small.py \
 && grep -q "forcings = forcings.copy()" graphcast_operational.py \
 && grep -q "forcings = forcings.copy()" graphcast_small.py

ENV PATH=/opt/envs/ui/bin:$PATH \
    DEMO3_ENV_ROOT=/opt/envs

WORKDIR /app
COPY . /app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501", \
     "--browser.gatherUsageStats", "false"]
