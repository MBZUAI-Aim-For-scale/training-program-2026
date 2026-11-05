#!/usr/bin/env bash
# Close everything down on the Spark: any setup run (including one started by a dashboard
# Launch button), the dashboard, and every running container.
#
#   bash demos/demo1/shutdown.sh
#
# Nothing is deleted: containers, images, saved runs and logs stay. Stopped containers
# don't come back after a reboot; start again with `spark_setup.py` or the dashboard.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

pkill -f 'python[0-9.]* \S*spark_setup\.py' && echo "setup run stopped"
python3 "$here/spark_setup.py" dashboard --stop
running=$(docker ps --format '{{.Names}}')
if [ -n "$running" ]; then
  docker stop $running >/dev/null && echo "containers stopped:" $running
else
  echo "no containers running"
fi
