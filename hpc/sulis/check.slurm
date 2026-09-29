#!/bin/bash
# Short verification job for Sulis (gpu-devel, <= 1 hour). Submit from the
# MSAGAT-Net checkout:
#     cd ~/msagat/MSAGAT-Net && sbatch hpc/sulis/check.slurm
# Establishes: modules + venv import, CUDA visible, data intact and identical
# across the three repos, test suite passes, and one representative run
# (LTLA, 372 nodes, the heaviest MSAGAT-Net cell) reproduces its local result
# and gives a wall-time to extrapolate campaign cost from.
#SBATCH --job-name=msagat-check
#SBATCH --account=su003-csmm
#SBATCH --partition=gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --time=00:55:00
#SBATCH --output=hpc/sulis/%x-%j.out
#SBATCH --export=ALL
set -euo pipefail

source hpc/sulis/env.sh
cd "$MSAGAT_ROOT/MSAGAT-Net"
echo "== $(date -u +%FT%TZ) host=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== imports and CUDA"
python - <<'EOF'
import torch, numpy, pandas, scipy, sklearn, tensorboardX
assert torch.cuda.is_available(), 'CUDA not visible'
x = torch.randn(4096, 4096, device='cuda')
print('torch', torch.__version__, '| device', torch.cuda.get_device_name(0),
      '| matmul ok', float((x @ x).abs().mean()) > 0)
EOF

echo "== data: the baselines read byte-identical copies of MSAGAT-Net's inputs"
# Exactly the series/adjacency pairs baseline_campaign.py uses (DATASETS);
# the repos also hold extras (threshold variants, spain-*, duplicates).
USED="japan.txt japan-adj.txt region785.txt region-adj.txt state360.txt
      state-adj-49.txt australia-covid.txt australia-adj.txt nhs_timeseries.txt
      nhs-adj.txt ltla_timeseries.txt ltla-adj.txt"
for repo in colagnn EpiGNN; do
    for f in $USED; do cmp -s "data/$f" "$MSAGAT_ROOT/$repo/data/$f" \
        || { echo "MISMATCH $repo/data/$f"; exit 1; }; done
    echo "$repo: all 12 files identical"
done

echo "== test suite"
# test_corpus_is_not_empty guards the token tests, which are generated from
# report/predictions/ archives; a fresh checkout has none (git-ignored), so
# that guard fails by design here. Everything else runs.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests \
    --deselect tests/test_tokens.py::test_corpus_is_not_empty

echo "== representative GPU run (reference: local RTX 5060 Ti, RMSE 87.6404, 134.6 s)"
REF=report/manifests/ltla_timeseries/MSAGAT-Net.ltla_timeseries.w-20.h-7.none.seed-42.with_adj.ltla-adj-100.loggrowth.quant.json
git show "HEAD:$REF" > /tmp/ref_manifest.json   # from git: a killed earlier check may have overwritten the file
python -m src.train --single --dataset ltla_timeseries \
    --horizon 7 --seed 42 --ablation none --pprm_supervision repeat \
    --target_space loggrowth --sim_mat ltla-adj-100 --quantiles | tail -5
python - "$REF" <<'EOF'
import json, sys
ref = json.load(open('/tmp/ref_manifest.json'))
new = json.load(open(sys.argv[1]))
r, n = ref['metrics']['rmse'], new['metrics']['rmse']
print(f"RMSE local {r:.4f} | sulis {n:.4f} | diff {100 * (n - r) / r:+.2f}%")
print(f"wall local {ref['wall_seconds']} s | sulis {new['wall_seconds']} s | "
      f"epochs local {ref['training']['epochs_run']} sulis {new['training']['epochs_run']}")
EOF

echo "== NHS cell on CPU only (is a GPU needed at all? local GPU mean for NHS: 56 s)"
SECONDS=0
python -m src.train --single --cpu --dataset nhs_timeseries \
    --horizon 7 --seed 42 --ablation none --pprm_supervision repeat \
    --target_space loggrowth --quantiles | tail -3
echo "CPU wall ${SECONDS} s (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo "== baselines on GPU: one short run per codebase, via baseline_campaign's own command builder"
# Reference test RMSEs are the seed-42 archives from the local RTX 5060 Ti
# campaign. lstnet stands in for CNNRNN_Res and dcrnn (same colagnn trainer);
# dcrnn itself takes 33-70 min even on the small sets, too long for gpu-devel.
python - <<'EOF'
import os, subprocess, time
import numpy as np
from src.scripts.baseline_campaign import build_cmd, npz_path

ADJ = {'japan': 'japan-adj', 'australia-covid': 'australia-adj'}
REF = [('epignn', 'japan', 3, 1272.3615, 49),
       ('cola_gnn', 'australia-covid', 3, 252.4466, 204),
       ('lstnet', 'australia-covid', 7, 269.3554, 33)]
for model, ds, h, ref_rmse, ref_s in REF:
    cwd, cmd = build_cmd(model, ds, ADJ[ds], h, 42)
    t0 = time.time()
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    wall = time.time() - t0
    if out.returncode != 0:
        print(out.stdout[-2000:], out.stderr[-2000:])
        raise SystemExit(f'{model} failed rc={out.returncode}')
    z = np.load(npz_path(model, ds, h, 42))
    rmse = float(np.sqrt(np.mean((z['y_true'] - z['y_pred']) ** 2)))
    print(f'{model:9s} {ds} h={h}: RMSE local {ref_rmse:.4f} | sulis {rmse:.4f} '
          f'({100 * (rmse - ref_rmse) / ref_rmse:+.2f}%) | wall local {ref_s} s, '
          f'sulis {wall:.0f} s')
EOF

# The check runs rewrote tracked manifests and all_results.csv rows in this
# checkout; restore them so the checkout matches the commit again.
git checkout -- report/manifests report/results
rm -f report/manifests/nhs_timeseries/MSAGAT-Net.nhs_timeseries.w-20.h-7.none.seed-42.with_adj.loggrowth.quant.json
git status --short
echo "== done $(date -u +%FT%TZ)"
