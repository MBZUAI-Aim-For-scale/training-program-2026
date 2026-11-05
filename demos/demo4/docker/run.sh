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

# ----------------------------------------------------------------------
# 1. Automated .env File Validation
# ----------------------------------------------------------------------
validate_env_file() {
  if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ENV_EXAMPLE" ]; then
      echo "[WARNING] .env file not found. Auto-generating .env from .env.example..."
      cp "$ENV_EXAMPLE" "$ENV_FILE"
    else
      echo "[ERROR] Error: Neither .env nor .env.example found at $root" >&2
      exit 1
    fi
  fi

  set -o allexport
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +o allexport

  missing_vars=()
  [ -z "${DEMO_BUCKET:-}" ] && missing_vars+=("DEMO_BUCKET")
  [ -z "${DEMO3_BUCKET:-}" ] && missing_vars+=("DEMO3_BUCKET")

  if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "[ERROR] Error: Missing required key(s) in .env file: ${missing_vars[*]}" >&2
    echo "        Please populate them in $ENV_FILE before proceeding." >&2
    exit 1
  fi

  echo "[OK] Configuration validated successfully."
}

# ----------------------------------------------------------------------
# 2. Host Environment Setup & Automatic Container Pruning
# ----------------------------------------------------------------------
validate_env_file

export OUTPUTS="${DEMO_OUTPUTS:-$here/outputs}"
export CACHE="${DEMO_CACHE:-$HOME/.cache/ai_weather}"
export ROMP_OUT="${ROMP_DEMO_OUT_DIR:-$here/ai_weather/data/ROMP_OUT}"

mkdir -p "$OUTPUTS" "$CACHE" "$ROMP_OUT"

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
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --wait --wait-timeout 120
  echo "[SUCCESS] Streamlit is healthy on the remote host: http://127.0.0.1:8503"
  echo "[INFO] SSH tunnel: ssh -N -L 8503:127.0.0.1:8503 <you>@country-spark"
}

# ----------------------------------------------------------------------
# 3. Command Router
# ----------------------------------------------------------------------
case "${1:-start}" in
  start)
    start_compose
    ;;

  logs)
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" logs -f
    ;;

  restart)
    echo "Restarting AI Weather services..."
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" restart
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --wait --wait-timeout 120
    echo "[SUCCESS] Services restarted."
    ;;

  update)
    echo "Pulling latest git updates..."
    git -C "$here" pull
    if [ "${DEV:-0}" = "1" ]; then
      echo "DEV=1 enabled; recreating services with live code mounts..."
      docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --wait --wait-timeout 120
    else
      echo "Updating container with incremental build cache..."
      docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --build --wait --wait-timeout 120
    fi
    ;;

  rebuild)
    echo "Rebuilding image from scratch..."
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" build --no-cache
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --force-recreate --wait --wait-timeout 120
    ;;

  down|stop)
    echo "Stopping AI Weather services..."
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down
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
