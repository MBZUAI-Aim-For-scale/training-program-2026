#!/usr/bin/env bash
# Start JupyterLab for the intro-to-Spark labs on a DGX Spark.
#   bash demos/intro-to-spark/docker/run.sh [port]   one person's or team's own, in this terminal (Lab 00):
#                                                    each uses their own port, and a new token each time
#   bash demos/intro-to-spark/docker/run.sh start    in the background, for the AI Weather Lab dashboard's
#                                                    "Intro to Spark" (Demo 1's spark_setup.py runs it, with
#                                                    PORT from demos/demo1/demos.toml: 8506). Its token stays
#                                                    the same, in ~/.cache/spark-intro/jupyter.env
#   bash demos/intro-to-spark/docker/run.sh stop     stop that one
# Ports in use on a Spark (demos/demo1/demos.toml lists them): 8500 the dashboard, 8502-8506 the demos,
# 8765 AI Almanac, 8888 Demo 4's Jupyter, 8889 Demo 1's sandbox. A person's own port: another one, e.g. 8890+.
# Jupyter binds to the Spark's localhost only; reach it by SSH tunnel. GPU=0: start it without the GPU.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${SPARK_INTRO_DATA:-${HOME}/.cache/spark-intro}"   # model weights (~35 GB) and LLM; survives restarts
mkdir -p "$DATA"
gpu=(--device nvidia.com/gpu=all)
[ "${GPU:-1}" = 1 ] || gpu=()
# --memory is deliberately not set: on Spark it caps CPU RAM only, not GPU memory (Lab 01).
common=(${gpu[@]+"${gpu[@]}"} --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
        -v "${HERE}:/workspace/intro-to-spark" -v "${DATA}:/workspace/data")
jupyter=(jupyter lab --ip 0.0.0.0 --allow-root --no-browser --ServerApp.root_dir=/workspace/intro-to-spark)

case "${1:-}" in
  start)
    # The dashboard's own: one container (spark-intro) on demos.toml's port, with a token kept in a file
    # the dashboard reads to open it signed in, inside its own page (Jupyter lets that page frame it).
    PORT="${PORT:-8506}"
    DASHBOARD_PORT="${DASHBOARD_PORT:-8500}"
    env_file="$DATA/jupyter.env"
    if [ ! -s "$env_file" ]; then
      (umask 077 && echo "JUPYTER_TOKEN=$(openssl rand -hex 16)" >"$env_file")
    fi
    TOKEN=$(sed -n 's/^JUPYTER_TOKEN=//p' "$env_file")
    frame="frame-ancestors 'self' http://localhost:${DASHBOARD_PORT} http://127.0.0.1:${DASHBOARD_PORT}"
    docker rm -f spark-intro >/dev/null 2>&1 || true
    docker run -d --name spark-intro "${common[@]}" -p "127.0.0.1:${PORT}:8888" spark-intro \
      "${jupyter[@]}" "--IdentityProvider.token=${TOKEN}" \
      "--ServerApp.tornado_settings={\"headers\": {\"Content-Security-Policy\": \"${frame}\"}}" >/dev/null
    echo "Intro to Spark: http://localhost:${PORT}/lab (the AI Weather Lab dashboard opens it signed in)"
    exit 0
    ;;
  stop)
    docker stop spark-intro >/dev/null && echo "stopped"
    exit 0
    ;;
esac

# One person's or team's own JupyterLab, in this terminal (Lab 00).
PORT="${1:-8890}"
TOKEN="${JUPYTER_TOKEN:-$(openssl rand -hex 8)}"

echo "Jupyter token: ${TOKEN}"
echo "On your laptop: ssh -N -L ${PORT}:localhost:${PORT} <spark>   then open http://localhost:${PORT}"

docker run --rm -it "${common[@]}" \
  --name "spark-intro-${USER}-${PORT}" \
  -p "127.0.0.1:${PORT}:8888" \
  -e JUPYTER_TOKEN="${TOKEN}" \
  spark-intro \
  "${jupyter[@]}" "--IdentityProvider.token=${TOKEN}"
