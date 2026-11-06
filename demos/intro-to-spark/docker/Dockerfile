# Image for the intro-to-Spark labs on DGX Spark (aarch64, GB10 Blackwell).
# Build from the repo root (~30-60 min the first time):
#   docker build -t spark-intro demos/intro-to-spark/docker
#
# These are the install steps of the image the labs were tested in (all 11 labs pass on a GB10 Spark,
# 2026-10-30). Some extras are not used by the labs; they are kept so the tested environment is unchanged.
FROM nvcr.io/nvidia/pytorch:26.06-py3

ENV EARTH2STUDIO_CACHE=/workspace/data/earth2cache \
    EARTH2STUDIO_PACKAGE_TIMEOUT=1200 \
    HF_HOME=/workspace/data/huggingface \
    NGC_CLI_ORG=no-org \
    NGC_CLI_TEAM=no-team \
    LOGURU_LEVEL=INFO \
    DEBIAN_FRONTEND=noninteractive \
    NATTEN_CUDA_ARCH=12.1 \
    NATTEN_N_WORKERS=8 \
    TORCH_CUDA_ARCH_LIST=12.1

ARG E2S_TAG=d654ad243d54a69b3d0251275dc29e0c96d402a1
ARG E2S_EXTRAS=aifsens,aurora,cosmo,da-cosmo,da-interp,data,sfno,statistics,stormcast-conus,stormscope,utils

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

# An aarch64 natten wheel exists on whl.natten.org; the source build is a fallback.
RUN uv pip install --system --break-system-packages "natten==0.21.7" -f https://whl.natten.org \
 || uv pip install --system --break-system-packages --no-build-isolation "natten==0.21.7"

RUN uv pip install --system --break-system-packages --upgrade ipywidgets jupyterlab nbclient nbformat

# earth2grid and torch-harmonics (pulled in by sfno) build extensions that import torch in setup.py;
# --no-build-isolation-package lets them see the base image's torch, which has the Blackwell kernels.
RUN uv pip install --system --break-system-packages \
    --no-build-isolation-package earth2grid \
    --no-build-isolation-package torch-harmonics \
    "earth2studio[${E2S_EXTRAS}] @ git+https://github.com/NVIDIA/earth2studio.git@${E2S_TAG}"

RUN uv pip install --system --break-system-packages cartopy seaborn windpowerlib==0.2.2

# Local LLM for Lab 05. Pinning huggingface-hub keeps earth2studio's version, so nothing else (torch included) changes.
RUN uv pip install --system --break-system-packages transformers accelerate "huggingface-hub==1.18.0"
RUN uv pip uninstall --system --break-system-packages jupyterlab-code-formatter jupyterlab-tensorboard-pro || true

RUN python -c "import earth2studio, cupy, sklearn, transformers; print('earth2studio', earth2studio.__version__)"

WORKDIR /workspace/intro-to-spark/labs
