#!/usr/bin/env bash
# Start forecasting and benchmarking Demos using docker compose and root .env config.
#
#   bash demos/demo4/docker/run.sh                start using docker compose
#   DEV=1 bash demos/demo4/docker/run.sh          enable live code mounting
#   bash demos/demo4/docker/run.sh logs           follow container logs
#   bash demos/demo4/docker/run.sh restart        restart running services
#   bash demos/demo4/docker/run.sh update         git pull and incrementally update container
#   bash demos/demo4/docker/run.sh rebuild        rebuild image from scratch and recreate container
#   bash demos/demo4/docker/run.sh down           stop and remove container services

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Locate compose file (check demos/demo4 or its docker/ folder)
if [ -f "$here/docker-compose.yml" ]; then
  COMPOSE_FILE="$here/docker-compose.yml"
elif [ -f "$here/docker/docker-compose.yml" ]; then
  COMPOSE_FILE="$here/docker/docker-compose.yml"
else
  echo "[ERROR] Could not find docker-compose.yml in $here or $here/docker" >&2
  exit 1
fi

# .env and .env.example stay at the repo root
root="$(cd "$here/../.." && pwd)"
ENV_FILE="$root/.env"
ENV_EXAMPLE="$root/.env.example"

# The program's bucket, used when .env doesn't set them (or there's no .env at all): a fresh
# clone on a Spark works with no manual step, reading the bucket with the Spark's key.
DEFAULT_DEMO_BUCKET="aim4scale_training_2026"
DEFAULT_DEMO3_BUCKET="gs://aim4scale_training_2026/demo3"

# ----------------------------------------------------------------------
# 1. Settings: .env if there is one (optional), defaults for the rest
# ----------------------------------------------------------------------
ENV_ARGS=()
validate_env_file() {
  if [ -f "$ENV_FILE" ]; then
    set -o allexport
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +o allexport
    ENV_ARGS=(--env-file "$ENV_FILE")
  else
    echo "[INFO] No .env at $root: using the defaults (see $ENV_EXAMPLE to change them)."
  fi

  # (Empty counts as unset, as in a .env copied from .env.example.)
  export DEMO_BUCKET="${DEMO_BUCKET:-$DEFAULT_DEMO_BUCKET}"
  export DEMO3_BUCKET="${DEMO3_BUCKET:-$DEFAULT_DEMO3_BUCKET}"
  echo "[OK] Bucket: gs://${DEMO_BUCKET#gs://}"
}

# ----------------------------------------------------------------------
# 2. Host Environment Setup & Automatic Container Pruning
# ----------------------------------------------------------------------
validate_env_file

export OUTPUTS="${DEMO_OUTPUTS:-$here/outputs}"
export CACHE="${DEMO_CACHE:-$HOME/.cache/ai_weather}"
export ROMP_OUT="${ROMP_DEMO_OUT_DIR:-$here/ai_weather/data/ROMP_OUT}"

mkdir -p "$OUTPUTS" "$CACHE" "$ROMP_OUT"

# This Spark's country data (Demo 1's country-data button), mounted read-only at /country_data:
# files there are read first, the bucket for the rest (ai_weather/config.py). Nothing if there's none.
cd_dir="${DEMO_COUNTRY_DATA_DIR:-$(python3 "$root/demos/demo1/country_data.py" --path 2>/dev/null || true)}"
if [ -n "$cd_dir" ] && [ -d "$cd_dir" ]; then export COUNTRY_DATA="$cd_dir"; else export COUNTRY_DATA=""; fi

FLEET_KEY="/etc/fleet-gcs-key.json"
ADC="$HOME/.config/gcloud/application_default_credentials.json"

export GCP_KEY_PATH=""
export GOOGLE_APPLICATION_CREDENTIALS=""

if [ -n "${DEMO3_GCP_KEY:-}" ] && [ -f "$DEMO3_GCP_KEY" ]; then
  export GCP_KEY_PATH="$DEMO3_GCP_KEY"
  export GOOGLE_APPLICATION_CREDENTIALS="/run/gcp-key.json"
elif [ -f "$FLEET_KEY" ]; then
  export GCP_KEY_PATH="$FLEET_KEY"
  export GOOGLE_APPLICATION_CREDENTIALS="/run/gcp-key.json"
elif [ -f "$ADC" ]; then
  export GCP_KEY_PATH="$ADC"
  export GOOGLE_APPLICATION_CREDENTIALS="/run/gcp-adc.json"
fi

check_gpu_availability() {
  echo "Checking GPU availability..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[OK] Host GPU detected:"
    nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader
  else
    echo "[WARNING] 'nvidia-smi' not found on host path."
  fi
}

start_compose() {
  check_gpu_availability
  echo "Starting AI Weather services via Docker Compose..."
  docker compose ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -f "$COMPOSE_FILE" up -d
  echo "[SUCCESS] AI Forecasting started: http://localhost:${PORT:-8503}"
  echo "[INFO] From another machine: ssh -N -L ${PORT:-8503}:127.0.0.1:${PORT:-8503} <you>@<country>-spark"
}

# ----------------------------------------------------------------------
# 3. Command Router
# ----------------------------------------------------------------------
case "${1:-start}" in
  start)
    start_compose
    ;;

  logs)
    docker compose ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -f "$COMPOSE_FILE" logs -f
    ;;

  restart)
    echo "Restarting AI Weather services..."
    docker compose ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -f "$COMPOSE_FILE" restart
    echo "[SUCCESS] Services restarted."
    ;;

  update)
    echo "Pulling latest git updates..."
    git -C "$here" pull
    if [ "${DEV:-0}" = "1" ]; then
      echo "DEV=1 enabled; restarting services..."
      docker compose ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -f "$COMPOSE_FILE" restart
    else
      echo "Updating container with incremental build cache..."
      docker compose ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -f "$COMPOSE_FILE" up -d --build
    fi
    ;;

  rebuild)
    echo "Rebuilding image from scratch..."
    docker compose ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -f "$COMPOSE_FILE" build --no-cache
    docker compose ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -f "$COMPOSE_FILE" up -d --force-recreate
    ;;

  down|stop)
    echo "Stopping AI Weather services..."
    docker compose ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -f "$COMPOSE_FILE" down
    ;;

  -h|--help|help)
    echo "Usage: bash demos/demo4/docker/run.sh [start|logs|restart|update|rebuild|down]"
    exit 0
    ;;

  *)
    echo "Error: Unknown command '$1'" >&2
    echo "Usage: bash demos/demo4/docker/run.sh [start|logs|restart|update|rebuild|down]" >&2
    exit 1
    ;;
esac
