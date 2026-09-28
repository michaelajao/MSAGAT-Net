# Source this (never execute it) on Sulis, on a login node or inside a job:
#     source ~/msagat/MSAGAT-Net/hpc/sulis/env.sh
# Modules must be loaded BEFORE the venv is activated, in every shell and every
# Slurm job, otherwise the venv's python cannot find the module-provided torch.
#
# Toolchain: GCC 13.2.0 (EasyBuild 2023b) + the Sulis-recommended PyTorch module.
# SciPy-bundle/2023.11 is the same toolchain generation and supplies numpy,
# scipy and pandas built against the module Python (3.11.5, matching dl_env's 3.11).
# Verify with `module spider PIP-PyTorch/2.4.0-CUDA-12.4.0` if a load fails.

module purge
module load GCC/13.2.0 OpenMPI/4.1.6
module load PIP-PyTorch/2.4.0-CUDA-12.4.0
module load SciPy-bundle/2023.11

export MSAGAT_ROOT="${MSAGAT_ROOT:-$HOME/msagat}"
source "$MSAGAT_ROOT/venv/bin/activate"

# Unbuffered stdout so Slurm logs show progress as it happens.
export PYTHONUNBUFFERED=1
# Each training process is tiny; stop BLAS/OpenMP from grabbing every core
# when several processes share a node.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
