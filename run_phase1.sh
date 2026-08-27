#!/usr/bin/env bash
# Phase 1 GPU campaigns, run one after the other.
# campaign.py launches one fresh subprocess per run and is serial by
# design, which is what the shared 16 GB card needs: an earlier
# baseline campaign at --parallel 2 lost three multi-hour LTLA jobs
# to OOM.
set -u
PY="$HOME/miniconda3/envs/dl_env/python.exe"
cd "$(dirname "$0")"
for chunk in ablation_v2 sensitivity_v2; do
  echo "=== $chunk starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$PY" -m src.scripts.campaign --chunk "$chunk"
  echo "=== $chunk finished rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done
echo "PHASE 1 CAMPAIGNS COMPLETE"
