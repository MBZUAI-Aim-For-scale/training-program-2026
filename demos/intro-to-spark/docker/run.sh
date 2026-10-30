#!/usr/bin/env bash
# Start JupyterLab for the intro-to-Spark labs on a DGX Spark.
#   bash demos/intro-to-spark/docker/run.sh [port]
# Each person/team uses their own port. Jupyter binds to the Spark's localhost only; reach it by SSH tunnel.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-8890}"   # 8888 is Demo 1's JupyterLab sandbox, 8500-8502 the demo dashboards
TOKEN="${JUPYTER_TOKEN:-$(openssl rand -hex 8)}"
DATA="${SPARK_INTRO_DATA:-${HOME}/.cache/spark-intro}"   # model weights (~35 GB) and LLM; survives restarts
mkdir -p "$DATA"

echo "Jupyter token: ${TOKEN}"
echo "On your laptop: ssh -N -L ${PORT}:localhost:${PORT} <spark>   then open http://localhost:${PORT}"

# --memory is deliberately not set: on Spark it caps CPU RAM only, not GPU memory (Lab 01).
docker run --rm -it --device nvidia.com/gpu=all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --name "spark-intro-${USER}-${PORT}" \
  -p "127.0.0.1:${PORT}:8888" \
  -v "${HERE}:/workspace/intro-to-spark" \
  -v "${DATA}:/workspace/data" \
  -e JUPYTER_TOKEN="${TOKEN}" \
  spark-intro \
  jupyter lab --ip 0.0.0.0 --allow-root --no-browser \
    --ServerApp.root_dir=/workspace/intro-to-spark --IdentityProvider.token="${TOKEN}"
