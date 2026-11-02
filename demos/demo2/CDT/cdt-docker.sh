#!/bin/bash
# Run the CDT GUI in Docker on a headless host.
#
#   ./cdt-docker.sh build          build the image
#   ./cdt-docker.sh [x11]          show the GUI on your ssh -X/-Y forwarded display
#   ./cdt-docker.sh vnc            serve the GUI in a browser via noVNC (port 6080)
#   ./cdt-docker.sh R              R console inside the container
#
# Environment:
#   CDT_DATA   host dir mounted at /data (CDT working dir)      default: $PWD
#   CDT_HOME   host dir for CDT's persistent config (~ in box)  default: ~/.cdt-docker
#   CDT_IMAGE  image name                                       default: cdt:latest
#   NOVNC_PORT port for vnc mode (bound to 127.0.0.1 only)      default: 6080
#   VNC_GEOMETRY  desktop size for vnc mode                     default: 1600x900
set -euo pipefail

IMAGE="${CDT_IMAGE:-cdt:latest}"
DATA="${CDT_DATA:-$PWD}"
CDT_HOME="${CDT_HOME:-$HOME/.cdt-docker}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
mode="${1:-x11}"; shift || true

if [ "$mode" = build ]; then
  exec docker build -t "$IMAGE" "$(dirname "$(readlink -f "$0")")"
fi

mkdir -p "$CDT_HOME"
args=(--rm -it
      --user "$(id -u):$(id -g)"
      -v "$CDT_HOME:/home/cdt"
      -v "$(readlink -f "$DATA"):/data")

case "$mode" in
  x11)
    if [ -z "${DISPLAY:-}" ]; then
      echo "No DISPLAY. Reconnect with 'ssh -Y user@host' (needs an X server locally:" >&2
      echo "XQuartz on macOS, VcXsrv/MobaXterm on Windows), or use: $0 vnc" >&2
      exit 1
    fi
    # Copy this session's X cookie into a hostname-independent cookie file.
    xauth_file="$CDT_HOME/.Xauthority-docker"
    : > "$xauth_file"
    xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$xauth_file" nmerge -
    args+=(--network host -e DISPLAY="$DISPLAY"
           -e XAUTHORITY=/home/cdt/.Xauthority-docker
           -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
    ;;
  vnc)
    args+=(-p "127.0.0.1:$NOVNC_PORT:6080" -e VNC_GEOMETRY="${VNC_GEOMETRY:-1600x900}")
    echo "On your local machine run:  ssh -L $NOVNC_PORT:localhost:$NOVNC_PORT $(whoami)@$(hostname)"
    echo "then open http://localhost:$NOVNC_PORT/vnc.html"
    ;;
esac

exec docker run "${args[@]}" "$IMAGE" "$mode" "$@"
