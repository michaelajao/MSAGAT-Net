#!/bin/bash
# Run one MSAGAT-Net campaign chunk on Sulis (gpu partition). Usage, from the
# MSAGAT-Net checkout:
#     sbatch hpc/sulis/campaign.slurm v2_main
#     sbatch --time=04:00:00 hpc/sulis/campaign.slurm ablation_v2
# Chunks are listed in src/scripts/campaign.py (CHUNKS). The driver skips any
# run whose prediction archive already exists, so resubmitting the same chunk
# after a timeout or failure resumes where it stopped.
#
# campaign.py runs one training process at a time, so this job leaves most of
# an A100 idle; request as little time as the chunk needs.
#SBATCH --job-name=msagat-campaign
#SBATCH --account=su003-csmm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --time=06:00:00
#SBATCH --output=hpc/sulis/%x-%j.out
#SBATCH --export=ALL
set -euo pipefail

CHUNK="${1:?usage: sbatch hpc/sulis/campaign.slurm <chunk>}"
source hpc/sulis/env.sh
cd "$MSAGAT_ROOT/MSAGAT-Net"
echo "== $(date -u +%FT%TZ) chunk=$CHUNK host=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader

python -m src.scripts.campaign --chunk "$CHUNK" --dry-run | head -1
srun python -m src.scripts.campaign --chunk "$CHUNK"
echo "== finished $(date -u +%FT%TZ)"
