#!/usr/bin/env bash
# docker build -t ai_weather -f docker/Dockerfile .
# Start forecasting and benchmarking Demos on a Spark from the image built by docker/Dockerfile.
#
#   bash docker/run.sh                 start using code baked into the image
#   DEV=1 bash docker/run.sh           use code in this folder; edits show up live
#   bash docker/run.sh logs            follow container logs
#   bash docker/run.sh restart         restart the existing container
#   bash docker/run.sh update          git pull; live with DEV=1, otherwise rebuild
#   bash docker/run.sh rebuild         rebuild image, recreate container, start it
#
# Then open http://localhost:8501 on the Spark
# (or tunnel: ssh -L 8501:localhost:8501 <spark>).
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUTS="${DEMO_OUTPUTS:-$here/outputs}"                 # saved forecasts, movies, job logs
CACHE="${DEMO_CACHE:-$HOME/.cache/ai_weather}"           # model weights (~90 GB once all are fetched)
mkdir -p "$OUTPUTS" "$CACHE"

container_exists() {
  docker container inspect ai_weather >/dev/null 2>&1
}

start_container() {
  code=() envs=()
  [ "${DEV:-0}" = 1 ] && code=(-v "$here:/app")
  # .env files are kept out of the image (.dockerignore); the code reads them at run time
  if [ "${DEV:-0}" != 1 ]; then
    for f in .env ai_weather/.env; do
      [ -f "$here/$f" ] && envs+=(-v "$here/$f:/app/$f:ro")
    done
  fi
  # Shared runs are downloaded from DEMO3_BUCKET (DEMO3_BUCKET= turns that off); runs made here stay here
  cloud=(-e DEMO3_BUCKET="${DEMO3_BUCKET-gs://aim4scale_training_2026/demo3}")
  # Credentials: DEMO3_GCP_KEY=<service-account.json>, else the Spark's read-only key for the
  # bucket, else this user's `gcloud auth application-default login`
  FLEET_KEY=/etc/fleet-gcs-key.json
  ADC="$HOME/.config/gcloud/application_default_credentials.json"
  if [ -n "${DEMO3_GCP_KEY:-}" ]; then
    cloud+=(-v "$DEMO3_GCP_KEY:/run/gcp-key.json:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/run/gcp-key.json)
  elif [ -f "$FLEET_KEY" ]; then
    cloud+=(-v "$FLEET_KEY:/run/gcp-key.json:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/run/gcp-key.json)
  elif [ -f "$ADC" ]; then
    cloud+=(-v "$ADC:/run/gcp-adc.json:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/run/gcp-adc.json)
  fi

  if container_exists; then
    echo "Removing existing container: ai_weather"
    docker rm -f ai_weather
  fi

  docker run -d --name ai_weather --restart unless-stopped \
    --device nvidia.com/gpu=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 127.0.0.1:8501:8501 \
    "${code[@]}" "${envs[@]}" "${cloud[@]}" -v "$OUTPUTS:/app/outputs" -v "$CACHE:/root/.cache" \
    -e DEMO_IFS_SOURCE="${DEMO3_IFS_SOURCE:-aws}" \
    ai_weather

  echo "AI Forecasting starting: http://localhost:8501"
}

case "${1:-start}" in
  start)
    start_container
    ;;

  logs)
    docker logs -f ai_weather
    ;;

  restart)
    if container_exists; then
      docker restart ai_weather
      echo "AI Forecasting restarted: http://localhost:8501"
    else
      echo "No existing ai_weather container; starting a new one."
      start_container
    fi
    ;;

  update)
    git -C "$here" pull
    if [ "${DEV:-0}" = 1 ]; then
      echo "Code is mounted live with DEV=1; restarting container."
      if container_exists; then
        docker restart ai_weather
      else
        start_container
      fi
    else
      echo "Rebuilding image and restarting container."
      docker build -t ai_weather -f docker/Dockerfile "$here"
      start_container
    fi
    ;;

  rebuild)
    docker build -t ai_weather -f docker/Dockerfile "$here"
    start_container
    ;;

  *)
    echo "Usage: bash docker/run.sh [start|logs|restart|update|rebuild]"
    exit 1
    ;;
esac
