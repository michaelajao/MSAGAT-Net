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
#SBATCH --time=00:50:00
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
cp "$REF" /tmp/ref_manifest.json
/usr/bin/time -v python -m src.train --single --dataset ltla_timeseries \
    --horizon 7 --seed 42 --ablation none --pprm_supervision repeat \
    --target_space loggrowth --sim_mat ltla-adj-100 --quantiles \
    2> /tmp/time_gpu.txt | tail -5
grep -E "Elapsed|Maximum resident" /tmp/time_gpu.txt
python - "$REF" <<'EOF'
import json, sys
ref = json.load(open('/tmp/ref_manifest.json'))
new = json.load(open(sys.argv[1]))
r, n = ref['metrics']['rmse'], new['metrics']['rmse']
print(f"RMSE local {r:.4f} | sulis {n:.4f} | diff {100 * (n - r) / r:+.2f}%")
print(f"wall local {ref['wall_seconds']} s | sulis {new['wall_seconds']} s | "
      f"epochs local {ref['training']['epochs_run']} sulis {new['training']['epochs_run']}")
EOF

echo "== same cell on CPU only (is a GPU needed at all?)"
/usr/bin/time -v python -m src.train --single --cpu --dataset nhs_timeseries \
    --horizon 7 --seed 42 --ablation none --pprm_supervision repeat \
    --target_space loggrowth --quantiles 2> /tmp/time_cpu.txt | tail -3
grep -E "Elapsed|Maximum resident" /tmp/time_cpu.txt

# The check runs rewrote tracked manifests and all_results.csv rows in this
# checkout; restore them so the checkout matches the commit again.
git checkout -- report/manifests report/results
git status --short
echo "== done $(date -u +%FT%TZ)"
