#!/usr/bin/env bash
# Start the forecasting and benchmarking demos on a Spark (or any Docker host) from the image
# built by docker/Dockerfile.
#
#   docker build -t ai_weather -f docker/Dockerfile .      (or: bash docker/run.sh rebuild)
#
#   bash docker/run.sh                 start (code baked into the image)
#   DEV=1 bash docker/run.sh           start, using the code in this folder (edits show up live)
#   GPU=0 bash docker/run.sh           start without a GPU (benchmarking needs none)
#   bash docker/run.sh check           check that the benchmark data is reachable
#   bash docker/run.sh momp ARGS...    run run_momp.py inside the container, e.g.
#                                        bash docker/run.sh momp --mode det --model AIFS
#   bash docker/run.sh shell           open a shell inside the container (the uv .venv is active)
#   bash docker/run.sh logs            follow the container logs
#   bash docker/run.sh restart         restart the existing container (keeps its old settings)
#   bash docker/run.sh stop            stop and remove the container
#   bash docker/run.sh update          git pull; live with DEV=1, otherwise rebuild
#   bash docker/run.sh rebuild         rebuild the image, recreate the container, start it
#
# Then open http://localhost:8501  (from another machine: ssh -L 8501:localhost:8501 <spark>).
#
# Folders (host -> container). Change them with the variables in brackets.
#   ~/data                              -> /app/data                       [DEMO_DATA]
#       external/   benchmark inputs (CHIRPS, region mask, model forecasts)
#       ROMP_OUT/   benchmark results (kept on the host when the container is removed)
#   apps/forecasting/outputs            -> /app/apps/forecasting/outputs   [DEMO_OUTPUTS]
#   ~/.cache/ai_weather                 -> /root/.cache                    [DEMO_CACHE]  (model weights)
#
# Settings: put DEMO_BUCKET=<bucket-name>, ROMP_REGION=Ethiopia, ... in .env (one KEY=value per
# line, no quotes). Docker passes that file to the container as it is.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DEMO_DATA:-$HOME/data}"
OUTPUTS="${DEMO_OUTPUTS:-$here/apps/forecasting/outputs}"
CACHE="${DEMO_CACHE:-$HOME/.cache/ai_weather}"
GCS_KEY="${DEMO_GCS_KEY:-$HOME/.config/ai_weather/gcs-key.json}"   # optional service-account key
IFS_SOURCE="${DEMO3_IFS_SOURCE:-${DEMO_IFS_SOURCE:-aws}}"
NAME=ai_weather

mkdir -p "$DATA/external" "$DATA/ROMP_OUT" "$OUTPUTS" "$CACHE"

container_exists() {
  docker container inspect "$NAME" >/dev/null 2>&1
}

build_image() {
  docker build -t "$NAME" -f docker/Dockerfile "$here"
}

start_container() {
  local code=() gpu=() envfile=() creds=()

  if [ "${DEV:-0}" = 1 ]; then
    # Use the code in this folder, but keep the image's /app/.venv (an anonymous volume is filled
    # from the image, so the folder mount does not hide it).
    code=(-v "$here:/app" -v /app/.venv)
  fi
  if [ "${GPU:-1}" = 1 ]; then
    gpu=(--device nvidia.com/gpu=all)
  fi
  if [ -f "$here/.env" ]; then
    envfile=(--env-file "$here/.env")
  fi

  # Google login for gs:// inputs (only needed when something is read from the bucket):
  #   1. a service-account key file, if you have one, else
  #   2. a one-hour token from your own `gcloud auth login` (skip with NO_GCLOUD_TOKEN=1)
  if [ -f "$GCS_KEY" ]; then
    creds=(-v "$GCS_KEY:/secrets/gcs-key.json:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcs-key.json)
  elif [ "${NO_GCLOUD_TOKEN:-0}" != 1 ] && command -v gcloud >/dev/null 2>&1; then
    tok="$(gcloud auth print-access-token 2>/dev/null || true)"
    if [ -n "$tok" ]; then
      creds=(-e DEMO_GCS_ACCESS_TOKEN="$tok")
    fi
  fi

  if container_exists; then
    echo "Removing existing container: $NAME"
    docker rm -fv "$NAME" >/dev/null
  fi

  docker run -d --name "$NAME" --restart unless-stopped \
    ${gpu[@]+"${gpu[@]}"} --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 127.0.0.1:8501:8501 \
    ${code[@]+"${code[@]}"} \
    -v "$DATA:/app/data" \
    -v "$OUTPUTS:/app/apps/forecasting/outputs" \
    -v "$CACHE:/root/.cache" \
    ${envfile[@]+"${envfile[@]}"} \
    ${creds[@]+"${creds[@]}"} \
    -e DEMO_DATA_DIR=/app/data \
    -e DEMO_OUTPUTS=/app/apps/forecasting/outputs \
    -e DEMO_IFS_SOURCE="$IFS_SOURCE" \
    -e DEMO3_IFS_SOURCE="$IFS_SOURCE" \
    "$NAME" >/dev/null

  echo "Started $NAME:  http://localhost:8501"
  echo "Data folder   : $DATA   (inputs in external/, results in ROMP_OUT/)"
  echo "Next          : bash docker/run.sh check"
}

need_container() {
  if ! container_exists; then
    echo "The container is not running. Start it first:  bash docker/run.sh" >&2
    exit 1
  fi
}

case "${1:-start}" in
  start)
    start_container
    ;;

  check)
    need_container
    docker exec "$NAME" python ai_weather/config.py
    ;;

  momp)
    need_container
    shift
    if [ "$#" -eq 0 ]; then
      echo "Usage: bash docker/run.sh momp --mode det|prob [--model AIFS] [--spec my_spec.json]" >&2
      exit 1
    fi
    docker exec -it "$NAME" python run_momp.py "$@"
    ;;

  shell)
    need_container
    docker exec -it "$NAME" bash
    ;;

  logs)
    docker logs -f "$NAME"
    ;;

  restart)
    if container_exists; then
      docker restart "$NAME"
      echo "Restarted $NAME (same settings as before). To pick up new mounts or .env changes use: bash docker/run.sh"
    else
      echo "No existing $NAME container; starting a new one."
      start_container
    fi
    ;;

  stop)
    if container_exists; then
      docker rm -fv "$NAME" >/dev/null
      echo "Container removed. Your data in $DATA is untouched."
    else
      echo "No $NAME container to stop."
    fi
    ;;

  update)
    git -C "$here" pull
    if [ "${DEV:-0}" = 1 ]; then
      echo "Code is mounted live with DEV=1; recreating the container."
      start_container
    else
      echo "Rebuilding the image and restarting the container."
      build_image
      start_container
    fi
    ;;

  rebuild)
    build_image
    start_container
    ;;

  *)
    echo "Usage: bash docker/run.sh [start|check|momp ARGS|shell|logs|restart|stop|update|rebuild]"
    exit 1
    ;;
esac
