#!/usr/bin/env bash
# Demo 5 (ForecastCheck) on a Spark: ForecastCheck's own docker/run.sh, with this program's
# settings, so every command in the lesson works as it is:
#
#   bash demos/demo5/run.sh fetch --apply     (from ~/training-program-2026)
#   bash demos/demo5/run.sh pipeline
#   bash demos/demo5/run.sh start | status | logs [pipeline] | shell | stop | help ...
#
# - The code: demos/demo5/forecastcheck, ForecastCheck's repository, which
#   `python3 demos/demo1/spark_setup.py` clones and keeps up to date (`--only demo5` for just it).
# - The port: Demo 5's in demos/demo1/demos.toml (8505), not ForecastCheck's default 8502,
#   which is Demo 3's. PORT=... overrides it.
# - The data: ~/demo_data/d5-AI-Forest-test (fetched data and every result), outside the code,
#   so updating or re-cloning ForecastCheck never touches it; forecastcheck/data is a link to
#   it, so the lesson's `data/...` paths work there too. DATA=... overrides it. (A checkout
#   that already has a real data/ folder with data in it keeps using that.)
# - The Google key: the Spark's read-only key (/etc/fleet-gcs-key.json), unless
#   GOOGLE_APPLICATION_CREDENTIALS is set.
# - The country: forecastcheck/.deployment. If there's none yet, the Spark's own country when
#   ForecastCheck has it (senegal-spark: senegal; kenya, rwanda, sierra-leone likewise).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
fc="$here/forecastcheck"
repo="$(cd "$here/../.." && pwd)"

if [ ! -f "$fc/docker/run.sh" ]; then
  echo "ForecastCheck isn't here yet. Get it with:  python3 $repo/demos/demo1/spark_setup.py --only demo5" >&2
  exit 1
fi

if [ -z "${PORT:-}" ]; then  # (demos.toml is the one place the demos' ports are set)
  PORT="$(python3 -c 'import sys, tomllib
cfg = tomllib.load(open(sys.argv[1], "rb"))
print(next(d["port"] for d in cfg["demo"] if d["id"] == "demo5"))' "$repo/demos/demo1/demos.toml" 2>/dev/null || echo 8505)"
fi
export PORT

if [ -z "${DATA:-}" ]; then
  if [ -d "$fc/data" ] && [ ! -L "$fc/data" ] && [ -n "$(ls -A "$fc/data" 2>/dev/null)" ]; then
    DATA="$fc/data"   # (from before: results stay where they are)
    echo "Data: $DATA (move it to ~/demo_data/d5-AI-Forest-test to use the usual place)" >&2
  else
    DATA="$HOME/demo_data/d5-AI-Forest-test"
    mkdir -p "$DATA"
    # The link the lesson's paths use (data/ here is missing, a link, or an empty folder);
    # git leaves it out (this checkout's own exclude list).
    [ -L "$fc/data" ] || rmdir "$fc/data" 2>/dev/null || true
    ln -sfn "$DATA" "$fc/data"
    if [ -d "$fc/.git/info" ] && ! grep -qx '/data' "$fc/.git/info/exclude" 2>/dev/null; then
      echo '/data' >>"$fc/.git/info/exclude"
    fi
  fi
fi
export DATA

if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -r /etc/fleet-gcs-key.json ]; then
  export GOOGLE_APPLICATION_CREDENTIALS=/etc/fleet-gcs-key.json
fi

if [ ! -f "$fc/.deployment" ] && [ -z "${VERIFICATION_DEPLOYMENT:-}" ]; then
  case "$(hostname -s 2>/dev/null || hostname)" in
    senegal*) country=senegal ;;
    sierra-leone* | sierraleone*) country=sierra-leone ;;
    kenya*) country=kenya ;;
    rwanda*) country=rwanda ;;
    *) country="" ;;
  esac
  if [ -n "$country" ]; then
    echo "$country" >"$fc/.deployment"
    echo "Country: $country (this Spark's; to change it: echo <country> > $fc/.deployment)"
  fi
fi

exec bash "$fc/docker/run.sh" "$@"
