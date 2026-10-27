#!/usr/bin/env bash
# Build and start one container per Dockerfile given on the command line.
#
#   bash launch_dockerfiles.sh demos/demo3/docker/Dockerfile docker/Dockerfile
#   REBUILD=1 bash launch_dockerfiles.sh ...     rebuild images even if they exist
#   PORT_START=8601 bash launch_dockerfiles.sh   first host port to hand out (default 8501)
#
# Names: a Dockerfile in a `docker/` folder is named after the folder above it
# (demos/demo3/docker/Dockerfile -> demo3, docker/Dockerfile -> ai_weather), and that
# folder is the build context; otherwise the Dockerfile's own folder is used.
# Ports: each container gets the next free host port, mapped to the first EXPOSE
# in its Dockerfile (8501 if none), so containers never clash.
set -euo pipefail

[ $# -gt 0 ] || { sed -n '2,14p' "$0"; exit 1; }

CACHE="${DEMO_CACHE:-$HOME/.cache/ai_weather}"           # model weights, shared by all containers
mkdir -p "$CACHE"
port="${PORT_START:-8501}"
started=()
failed=()

port_in_use() {
  ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN
}

for dockerfile in "$@"; do
  if [ ! -f "$dockerfile" ]; then
    echo "!! $dockerfile: no such file"; failed+=("$dockerfile"); continue
  fi
  dockerfile="$(cd "$(dirname "$dockerfile")" && pwd)/$(basename "$dockerfile")"
  context="$(dirname "$dockerfile")"
  [ "$(basename "$context")" = docker ] && context="$(dirname "$context")"
  name="$(basename "$context" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_.\n-' '_')"

  inner="$(grep -im1 '^EXPOSE' "$dockerfile" | awk '{print $2}' | cut -d/ -f1 || true)"
  inner="${inner:-8501}"
  while port_in_use "$port"; do port=$((port + 1)); done

  echo "== $name  ($dockerfile)"
  if [ "${REBUILD:-0}" = 1 ] || ! docker image inspect "$name" >/dev/null 2>&1; then
    if ! docker build -t "$name" -f "$dockerfile" "$context"; then
      echo "!! $name: docker build failed"; failed+=("$name"); continue
    fi
  else
    echo "   image $name exists (REBUILD=1 to rebuild)"
  fi

  docker rm -f "$name" >/dev/null 2>&1 || true
  if docker run -d --name "$name" --restart unless-stopped \
      --device nvidia.com/gpu=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      -p "127.0.0.1:$port:$inner" -v "$CACHE:/root/.cache" \
      "$name" >/dev/null; then
    started+=("$name -> http://localhost:$port")
    port=$((port + 1))
  else
    echo "!! $name: docker run failed"; failed+=("$name")
  fi
done

echo
for s in "${started[@]}"; do echo "started  $s"; done
for f in "${failed[@]}"; do echo "FAILED   $f"; done
[ ${#failed[@]} -eq 0 ]
