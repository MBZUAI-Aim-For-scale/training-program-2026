#!/usr/bin/env bash
# ==============================================================================
# DOCKER RUN & REBUILD INSTRUCTIONS
# ==============================================================================
# This script wraps `docker compose` to manage the AI Weather container.
#
# HOW TO RUN:
#   bash docker/run.sh start             Start the container (uses baked-in code)
#   DEV=1 bash docker/run.sh start       Start with live code mounting (edits show up instantly)
#   bash docker/run.sh logs              Follow the container logs
#   bash docker/run.sh restart           Restart the existing container
#   bash docker/run.sh update            Git pull, then restart (or rebuild if DEV=0)
#   bash docker/run.sh rebuild           FORCE REBUILD: Rebuild the Docker image from scratch, recreate the container, and start it
#   bash docker/run.sh down              Stop and remove the container
#
# HOW TO REBUILD MANUALLY (without the script):
#   docker compose -f docker/docker-compose.yml build --no-cache
#   docker compose -f docker/docker-compose.yml up -d
#
# Then open http://localhost:8501 on the host machine.
# ==============================================================================

set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"

# ==============================================================================
# 1. Setup directories and environment variables for docker-compose
# ==============================================================================
OUTPUTS="${DEMO_OUTPUTS:-$here/outputs}"
CACHE="${DEMO_CACHE:-$HOME/.cache/ai_weather}"
mkdir -p "$OUTPUTS" "$CACHE"

export OUTPUTS_DIR="$OUTPUTS"
export CACHE_DIR="$CACHE"
export PROJECT_ROOT="$here"
export ENV_FILE_1="$here/.env"
export ENV_FILE_2="$here/ai_weather/.env"

# Ensure .env files exist so Docker Compose bind mounts don't fail
touch "$ENV_FILE_1" "$ENV_FILE_2" 2>/dev/null || true

# ==============================================================================
# 2. Handle GCP Credentials Fallback (Dynamic Override)
# ==============================================================================
FLEET_KEY=/etc/fleet-gcs-key.json
ADC="$HOME/.config/gcloud/application_default_credentials.json"
GCP_OVERRIDE="$here/docker/docker-compose.gcp.yml"

rm -f "$GCP_OVERRIDE"

if [ -n "${DEMO3_GCP_KEY:-}" ] && [ -f "$DEMO3_GCP_KEY" ]; then
  cat > "$GCP_OVERRIDE" <<EOF
services:
  ai_weather:
    volumes:
      - $DEMO3_GCP_KEY:/run/gcp-key.json:ro
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/run/gcp-key.json
EOF
elif [ -f "$FLEET_KEY" ]; then
  cat > "$GCP_OVERRIDE" <<EOF
services:
  ai_weather:
    volumes:
      - $FLEET_KEY:/run/gcp-key.json:ro
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/run/gcp-key.json
EOF
elif [ -f "$ADC" ]; then
  cat > "$GCP_OVERRIDE" <<EOF
services:
  ai_weather:
    volumes:
      - $ADC:/run/gcp-adc.json:ro
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/run/gcp-adc.json
EOF
fi

# ==============================================================================
# 3. Build the docker compose command
# ==============================================================================
COMPOSE_FILES=(-f docker/docker-compose.yml)

if [ -f "$GCP_OVERRIDE" ]; then
  COMPOSE_FILES+=(-f "$GCP_OVERRIDE")
fi

if [ "${DEV:-0}" = 1 ]; then
  COMPOSE_FILES+=(-f docker/docker-compose.dev.yml)
fi

# ==============================================================================
# 4. Command Router
# ==============================================================================
start_container() {
  echo "Starting AI Forecasting container via docker compose..."
  docker compose "${COMPOSE_FILES[@]}" up -d --build
  echo "AI Forecasting starting: http://localhost:8501"
}

case "${1:-start}" in
  start)
    start_container
    ;;
    
  logs)
    docker compose "${COMPOSE_FILES[@]}" logs -f ai_weather
    ;;
    
  restart)
    docker compose "${COMPOSE_FILES[@]}" restart ai_weather
    echo "AI Forecasting restarted: http://localhost:8501"
    ;;
    
  update)
    git -C "$here" pull
    if [ "${DEV:-0}" = 1 ]; then
      echo "Code is mounted live with DEV=1; restarting container."
      docker compose "${COMPOSE_FILES[@]}" restart ai_weather
    else
      echo "Rebuilding image and restarting container."
      docker compose "${COMPOSE_FILES[@]}" up -d --build
    fi
    ;;
    
  rebuild)
    echo "Force rebuilding image from scratch (ignoring cache)..."
    docker compose "${COMPOSE_FILES[@]}" build --no-cache
    docker compose "${COMPOSE_FILES[@]}" up -d
    echo "Rebuild complete. AI Forecasting starting: http://localhost:8501"
    ;;
    
  down)
    docker compose "${COMPOSE_FILES[@]}" down
    echo "Container stopped and removed."
    ;;
    
  *)
    echo "Usage: bash docker/run.sh [start|logs|restart|update|rebuild|down]"
    exit 1
    ;;
esac
