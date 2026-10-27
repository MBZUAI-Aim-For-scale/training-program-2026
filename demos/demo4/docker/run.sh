#!/usr/bin/env bash
# Start Demo 4 on a Spark from the image built by demos/demo4/docker/Dockerfile.
#   bash demos/demo4/docker/run.sh          code baked into the image (what each country's Spark runs)
#   DEV=1 bash demos/demo4/docker/run.sh    use the code in this repo instead, so edits show up live
# Then open http://localhost:8503 on the Spark (or tunnel: ssh -L 8503:localhost:8503 <spark>).
# Benchmarking needs no GPU, so the container gets none.
set -euo pipefail
here="$(cd "$(dirname "$0")/../../.." && pwd)"   # repo root
DATA="${DEMO_DATA:-$HOME/data}"                  # external/: benchmark inputs, ROMP_OUT/: results
mkdir -p "$DATA/external" "$DATA/ROMP_OUT"

code=()
# The repo replaces /app, but the image's /app/.venv stays (an anonymous volume is filled from the image)
[ "${DEV:-0}" = 1 ] && code=(-v "$here:/app" -v /app/.venv)
# Settings such as DEMO_BUCKET or ROMP_REGION go in the repo's .env (one KEY=value per line)
envfile=()
[ -f "$here/.env" ] && envfile=(--env-file "$here/.env")
# Credentials for gs:// inputs (config.py falls back to the bucket when local data is missing):
# DEMO4_GCP_KEY=<service-account.json>, else the Spark's read-only key for the bucket, else this
# user's `gcloud auth application-default login`
cloud=()
FLEET_KEY=/etc/fleet-gcs-key.json
ADC="$HOME/.config/gcloud/application_default_credentials.json"
if [ -n "${DEMO4_GCP_KEY:-}" ]; then
  cloud=(-v "$DEMO4_GCP_KEY:/run/gcp-key.json:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/run/gcp-key.json)
elif [ -f "$FLEET_KEY" ]; then
  cloud=(-v "$FLEET_KEY:/run/gcp-key.json:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/run/gcp-key.json)
elif [ -f "$ADC" ]; then
  cloud=(-v "$ADC:/run/gcp-adc.json:ro" -e GOOGLE_APPLICATION_CREDENTIALS=/run/gcp-adc.json)
fi

docker run -d --name demo4 --restart unless-stopped \
  -p 127.0.0.1:8503:8501 \
  ${code[@]+"${code[@]}"} ${envfile[@]+"${envfile[@]}"} ${cloud[@]+"${cloud[@]}"} \
  -v "$DATA:/app/data" -e DEMO_DATA_DIR=/app/data \
  demo4
echo "Demo 4 starting: http://localhost:8503"
