#!/usr/bin/env bash
# Phase 1 GPU campaigns, serial. The 16 GB card is shared with the desktop,
# so --parallel 1 is required: an earlier run at --parallel 2 lost three
# multi-hour LTLA jobs to OOM.
set -u
PY="$HOME/miniconda3/envs/dl_env/python.exe"
cd "$(dirname "$0")"
for chunk in ablation_v2 sensitivity_v2; do
  echo "=== $chunk starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$PY" -m src.scripts.campaign --chunk "$chunk" --parallel 1
  echo "=== $chunk finished rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done
echo "PHASE 1 CAMPAIGNS COMPLETE"
