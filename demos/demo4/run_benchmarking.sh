#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ../../.venv/bin/activate

cd ../..

#bash docker/run.sh --rebuild

python run_momp.py --mode det
