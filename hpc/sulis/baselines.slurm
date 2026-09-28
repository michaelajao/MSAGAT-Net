#!/bin/bash
# Retrain the five baselines under the corrected lead-h protocol on Sulis.
# Needs ../colagnn and ../EpiGNN next to MSAGAT-Net (baseline_campaign.py
# resolves them as siblings). Usage, from the MSAGAT-Net checkout:
#     sbatch hpc/sulis/baselines.slurm --models lstnet CNNRNN_Res epignn
#     sbatch --time=12:00:00 hpc/sulis/baselines.slurm --models cola_gnn dcrnn
# Extra arguments go straight to src.scripts.baseline_campaign. It skips runs
# whose archive exists, so resubmitting resumes.
#
# --parallel 6: the light baselines use well under 2 GB of GPU memory each, so
# six share one A100 40 GB. cola_gnn and dcrnn on LTLA took up to 10.3 h
# locally; give those their own job with a long --time.
#SBATCH --job-name=msagat-baselines
#SBATCH --account=su003-csmm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --time=08:00:00
#SBATCH --output=hpc/sulis/%x-%j.out
#SBATCH --export=ALL
set -euo pipefail

source hpc/sulis/env.sh
cd "$MSAGAT_ROOT/MSAGAT-Net"
echo "== $(date -u +%FT%TZ) args=$* host=$(hostname) job=$SLURM_JOB_ID"
echo "   MSAGAT-Net $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader

python -m src.scripts.baseline_campaign --dry-run "$@" | head -1
srun python -m src.scripts.baseline_campaign --parallel "${PARALLEL:-6}" "$@"
echo "== finished $(date -u +%FT%TZ)"
