#!/usr/bin/env bash
# Compatibility entrypoint for the relocated Demo 4 Docker stack.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/docker/run.sh" "$@"
