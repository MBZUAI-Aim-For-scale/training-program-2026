#!/usr/bin/env bash
# Start the CDT GUI (Climate Data Tools) on a Spark, served in the browser through noVNC.
#   bash docker/run.sh      image built from CDT/Dockerfile (spark_setup.py --only cdt builds it)
# Then open http://localhost:8504/vnc.html on the Spark (or tunnel: ssh -L 8504:localhost:8504 <spark>).
# PORT: the host port (spark_setup.py passes the one in demos/demo1/demos.toml; 8504 by hand).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8504}"
DATA="${CDT_DATA:-$here/data}"                     # CDT's working dir (/data in the container)
CDT_HOME="${CDT_HOME:-$HOME/.cdt-docker}"          # CDT's settings, kept between runs
mkdir -p "$DATA" "$CDT_HOME"

# Files CDT writes should belong to you. Rootless Docker already maps the container's root to you;
# otherwise run as your own user (as root it couldn't write to the folders above anyway).
user=()
docker info --format '{{.SecurityOptions}}' | grep -q rootless || user=(--user "$(id -u):$(id -g)")

docker run -d --name cdt --restart unless-stopped \
  "${user[@]}" \
  -p "127.0.0.1:$PORT:6080" \
  -v "$DATA:/data" -v "$CDT_HOME:/home/cdt" \
  -e VNC_GEOMETRY="${VNC_GEOMETRY:-1600x900}" \
  cdt vnc
echo "CDT starting: http://localhost:$PORT/vnc.html"
