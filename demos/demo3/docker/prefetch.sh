#!/usr/bin/env bash
# Download every Demo 3 model's weights (~21 GB) into the cache docker/run.sh mounts, so no model's
# first run in class waits for them. spark_setup.py runs this after building the image (demos.toml
# `prefetch`); it returns at once and the download carries on in the background, in the container
# demo3-weights (`docker logs -f demo3-weights`). Models already fetched are skipped, so running it
# again only fetches what's missing or failed.
#   bash docker/prefetch.sh          DEV=1: with this folder's code, as run.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${DEMO3_CACHE:-$HOME/.cache/demo3}"   # the same as run.sh's
NAME=demo3-weights
mkdir -p "$CACHE"

code=()
[ "${DEV:-0}" = 1 ] && code=(-v "$here:/app")

if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ]; then
  echo "still downloading the models' weights in the background (docker logs -f $NAME)"
  exit 0
fi
# (the image's entrypoint prints a banner first: the answer is the last line)
missing="$(docker run --rm ${code[@]+"${code[@]}"} -v "$CACHE:/root/.cache" demo3 \
  python runners/prefetch_weights.py --missing | tail -n 1)"
if [ -z "$missing" ]; then
  echo "every model's weights are here ($(du -sh "$CACHE" 2>/dev/null | cut -f1) in $CACHE)"
  exit 0
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true   # (the last download's finished container)
# (the GPU as run.sh gives it, so the models load as in a run; they load on the CPU, so it uses none)
docker run -d --name "$NAME" --device nvidia.com/gpu=all --ipc=host \
  ${code[@]+"${code[@]}"} -v "$CACHE:/root/.cache" -e PYTHONUNBUFFERED=1 \
  demo3 python runners/prefetch_weights.py >/dev/null
echo "downloading weights for ${missing// /, } in the background (docker logs -f $NAME)"
