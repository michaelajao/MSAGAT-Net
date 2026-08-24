#!/usr/bin/env bash
# Run the regression suite in dl_env, bypassing a broken third-party plugin.
set -euo pipefail
PY="${PY:-$HOME/miniconda3/envs/dl_env/python.exe}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest "$@"
