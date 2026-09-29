#!/bin/bash
# One-off: create the project venv on Sulis on top of the modules in env.sh.
# Run on a LOGIN node (pip needs the network; no GPU is needed to install):
#     bash ~/msagat/MSAGAT-Net/hpc/sulis/setup_venv.sh
# Safe to re-run: it refuses to overwrite an existing venv.
set -euo pipefail

MSAGAT_ROOT="${MSAGAT_ROOT:-$HOME/msagat}"
VENV="$MSAGAT_ROOT/venv"

module purge
module load GCC/13.2.0 OpenMPI/4.1.6
module load PIP-PyTorch/2.4.0-CUDA-12.4.0
module load SciPy-bundle/2023.11

if [ -d "$VENV" ]; then
    echo "venv already exists at $VENV; delete it first to rebuild" >&2
    exit 1
fi

# --system-site-packages: torch, numpy, scipy and pandas come from the modules
# (built for Sulis); only the pure-Python extras are pip-installed.
python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
# Pin the module-provided core so pip cannot shadow it with wheels built for
# another stack (versions as loaded on Sulis, checked 29 Sep 2026).
CONSTRAINTS="$MSAGAT_ROOT/constraints.txt"
printf 'torch==2.4.0\nnumpy==1.26.2\nscipy==1.11.4\npandas==2.1.3\n' > "$CONSTRAINTS"
# tensorboardX is imported by the colagnn and EpiGNN trainers; tensorboard by
# MSAGAT-Net's optional writer. colagnn's train.py imports
# spatiotemporal_transformer_gat at start-up, which needs torch_geometric's
# GATConv (pure Python; no torch_scatter/torch_sparse needed).
python -m pip install --constraint "$CONSTRAINTS" \
    scikit-learn==1.5.2 matplotlib==3.8.4 seaborn==0.13.2 \
    tensorboard==2.17.1 tensorboardX==2.6.2.2 pytest==7.4.4 \
    torch_geometric==2.6.1

python - <<'EOF'
import sys, torch, numpy, pandas, scipy, sklearn
print('python', sys.version.split()[0])
print('torch', torch.__version__, 'cuda build', torch.version.cuda)
print('numpy', numpy.__version__, 'pandas', pandas.__version__,
      'scipy', scipy.__version__, 'sklearn', sklearn.__version__)
EOF
python -m pip freeze > "$MSAGAT_ROOT/venv-freeze.txt"
echo "venv ready: $VENV (freeze saved to $MSAGAT_ROOT/venv-freeze.txt)"
