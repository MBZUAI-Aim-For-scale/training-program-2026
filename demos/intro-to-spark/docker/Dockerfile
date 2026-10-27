# Image for the intro-to-Spark labs (JupyterLab + Earth2Studio with SFNO and data sources).
# Build from the repo root:  docker build -t spark-intro demos/intro-to-spark/docker
FROM nvcr.io/nvidia/pytorch:26.06-py3

ENV EARTH2STUDIO_CACHE=/root/.cache/earth2studio \
    EARTH2STUDIO_PACKAGE_TIMEOUT=1200 \
    LOGURU_LEVEL=INFO \
    DEBIAN_FRONTEND=noninteractive

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

# Same Earth2Studio commit the labs were tested with. The base image's torch is kept (Blackwell kernels):
# never let pip replace it.
ARG E2S_TAG=d654ad243d54a69b3d0251275dc29e0c96d402a1
RUN uv pip install --system --break-system-packages \
    --no-build-isolation-package earth2grid \
    "earth2studio[sfno,data] @ git+https://github.com/NVIDIA/earth2studio.git@${E2S_TAG}" \
    jupyterlab ipywidgets matplotlib

RUN python -c "import torch, earth2studio; print('torch', torch.__version__, '| earth2studio', earth2studio.__version__)" \
 && python -c "from earth2studio.models.px import SFNO; from earth2studio.data import GFS, ARCO"

WORKDIR /workspace/intro-to-spark/labs
