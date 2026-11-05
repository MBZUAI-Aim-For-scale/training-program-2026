#!/usr/bin/env bash
# Demo 5 (ForecastCheck) on a Spark: ForecastCheck's own docker/run.sh, with this program's
# settings, so every command in the lesson works as it is:
#
# Demo 1's setup installs it as the command `forecastcheck` (~/.local/bin), which works the
# same from any folder; nothing here depends on the current directory.
#
#   forecastcheck setup senegal     the whole Section 3: country, fetch, pipeline, dashboard
#   forecastcheck country [NAME]    show or set the country (senegal, sierra-leone, kenya, rwanda)
#   forecastcheck where             where the code, the data and the settings are
#   forecastcheck fetch [--apply] | pipeline | start | status | logs [pipeline] | shell |
#                 stop | help ...   ForecastCheck's own commands (its docker/run.sh), as they are
#
# - The code: demos/demo5/forecastcheck, ForecastCheck's repository, which
#   `python3 demos/demo1/spark_setup.py` clones and keeps up to date (`--only demo5` for just it).
# - The port: Demo 5's in demos/demo1/demos.toml (8505), not ForecastCheck's default 8502,
#   which is Demo 3's. PORT=... overrides it.
# - The data: demos/demo5/data (fetched data and every result; git leaves it out), as the
#   other demos keep theirs in their folders. DATA=... overrides it.
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

# The data, like the other demos': in this demo's folder, next to (not inside) ForecastCheck's
# code, so updating or re-cloning ForecastCheck never touches it. (A checkout that already has
# results in its own data/, from before, keeps using that.)
if [ -z "${DATA:-}" ]; then
  if [ -d "$fc/data" ] && [ ! -L "$fc/data" ] && [ -n "$(ls -A "$fc/data" 2>/dev/null)" ]; then
    DATA="$fc/data"
  else
    DATA="$here/data"
  fi
fi
mkdir -p "$DATA"
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
    echo "Country: $country (this Spark's; to change it: forecastcheck country <name>)"
  fi
fi

countries() { (cd "$fc/deployments" && ls -- *.toml 2>/dev/null | sed 's/\.toml$//' | tr '\n' ' '); }
set_country() {
  [ -f "$fc/deployments/$1.toml" ] || { echo "No country '$1'. ForecastCheck has: $(countries)" >&2; exit 1; }
  echo "$1" >"$fc/.deployment"
  echo "Country: $1"
}
current_country() { grep -v '^[[:space:]]*\(#\|$\)' "$fc/.deployment" 2>/dev/null | head -1 | tr -d '[:space:]'; }

case "${1:-help}" in
  country)
    if [ -n "${2:-}" ]; then
      set_country "$2"
      # (a running dashboard follows a new country after a restart)
      if docker container inspect forecastcheck >/dev/null 2>&1; then exec bash "$fc/docker/run.sh" restart; fi
    else
      echo "Country: ${VERIFICATION_DEPLOYMENT:-$(current_country)}${VERIFICATION_DEPLOYMENT:+ (from VERIFICATION_DEPLOYMENT)}"
      [ -n "${VERIFICATION_DEPLOYMENT:-}$(current_country)" ] || echo "None yet: forecastcheck country <name>  ($(countries))"
    fi
    exit 0 ;;
  where)
    echo "code:      $fc"
    echo "data:      $DATA"
    echo "settings:  $fc/config/verification.toml"
    echo "countries: $fc/deployments/   (this one: $(current_country || true))"
    echo "dashboard: http://localhost:$PORT"
    exit 0 ;;
  setup)
    # The whole Section 3 in one go: [country], the data, the pipeline (options after the
    # country go to it, e.g. --full), and the dashboard.
    shift
    if [ -n "${1:-}" ] && [ "${1#-}" = "$1" ]; then set_country "$1"; shift; fi
    [ -n "${VERIFICATION_DEPLOYMENT:-}$(current_country)" ] ||
      { echo "Which country? forecastcheck setup <country>  ($(countries))" >&2; exit 1; }
    bash "$fc/docker/run.sh" fetch --apply
    bash "$fc/docker/run.sh" pipeline "$@" || echo "(no new pipeline started; see above)"
    exec bash "$fc/docker/run.sh" start ;;
esac

exec bash "$fc/docker/run.sh" "$@"
