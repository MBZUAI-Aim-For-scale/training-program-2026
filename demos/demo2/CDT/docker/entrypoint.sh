#!/bin/bash
# Modes:
#   x11  - draw on $DISPLAY (e.g. an `ssh -X` forwarded display)   [default]
#   vnc  - run a private X server and serve it with noVNC on :6080
#   R    - plain R console (any extra args are passed to R)
set -e
export HOME=/home/cdt
mkdir -p "$HOME/Documents"

# Start the GUI, then keep R alive until the CDT main window is closed.
START_CDT='library(CDT); startCDT(wd = getwd());
  win <- CDT:::.cdtEnv$tcl$main$win;
  while (tcltk::tclvalue(tcltk::tkwinfo("exists", win)) == "1") Sys.sleep(0.5)'

mode="${1:-x11}"; shift || true
case "$mode" in
  x11)
    if [ -z "$DISPLAY" ]; then
      echo "DISPLAY is not set. Connect with 'ssh -X' (or -Y) and use cdt-docker.sh." >&2
      exit 1
    fi
    exec Rscript -e "$START_CDT"
    ;;
  vnc)
    export DISPLAY=:1
    geometry="${VNC_GEOMETRY:-1600x900}"
    Xvnc :1 -geometry "$geometry" -depth 24 -SecurityTypes None \
         -localhost yes -rfbport 5901 -AlwaysShared >/tmp/xvnc.log 2>&1 &
    for _ in $(seq 50); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.1; done
    openbox >/tmp/openbox.log 2>&1 &
    websockify --web /usr/share/novnc "${NOVNC_PORT:-6080}" localhost:5901 >/tmp/novnc.log 2>&1 &
    echo "noVNC ready: http://localhost:${NOVNC_PORT:-6080}/vnc.html (tunnel this port over ssh)"
    # Restart CDT if the window is closed; stop the container with Ctrl-C / docker stop.
    while true; do Rscript -e "$START_CDT" || true; sleep 1; done
    ;;
  R)
    exec R "$@"
    ;;
  *)
    exec "$mode" "$@"
    ;;
esac
