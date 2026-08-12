"""Extract learned attention matrices from trained checkpoints.

For each MSAGAT-Net checkpoint, rebuilds the model, forwards the test split,
and saves the batch- and head-averaged attention matrix [N, N] to
report/attention/<token>.npy. These matrices are the calibration-transfer
kernels for the attention-coupled conformal layer.

Usage (from the repository root):
    python -m src.scripts.extract_attention --pattern .loggrowth.quant
"""

import argparse
import glob
import os
import re

import numpy as np
import torch

from ..data import DataBasicLoader
from ..models import MSAGATNet_Ablation
from ..train import DATASET_CONFIGS, DEFAULT_QUANTILES, TRAIN_DEFAULTS
from argparse import Namespace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(BASE_DIR, 'save_all')
OUT_DIR = os.path.join(BASE_DIR, 'report', 'attention')

TOKEN_RE = re.compile(
    r'^MSAGAT-Net\.(?P<ds>.+)\.w-20\.h-(?P<h>\d+)\.(?P<abl>[a-z_]+)'
    r'\.seed-(?P<s>\d+)\.with_adj(?P<variant>(\.[A-Za-z0-9-]+)*)\.pt$')


def build_args(dataset, horizon, variant):
    cfg = DATASET_CONFIGS[dataset]
    return Namespace(
        dataset=dataset, sim_mat=cfg['sim_mat'],
        window=TRAIN_DEFAULTS['window'], horizon=horizon,
        train=0.6, val=0.2, test=0.2,
        batch=TRAIN_DEFAULTS['batch'], dropout=TRAIN_DEFAULTS['dropout'],
        ablation='none', hidden_dim=TRAIN_DEFAULTS['hidden_dim'],
        attention_heads=TRAIN_DEFAULTS['attention_heads'],
        attention_regularization_weight=1e-5,
        num_scales=TRAIN_DEFAULTS['num_scales'], kernel_size=3,
        feature_channels=16, bottleneck_dim=TRAIN_DEFAULTS['bottleneck_dim'],
        use_adj_prior=True, adj_weight=0.1, use_graph_bias=True,
        adaptive=False, gpu=0, cuda=torch.cuda.is_available(),
        save_dir='save_all', mylog=False, highway_window=4,
        extra='', label='', pcc='',
        pprm_supervision='repeat',
        spatial_gate='.sgate' in variant,
        target_space='loggrowth' if '.loggrowth' in variant else 'level',
        quantiles=DEFAULT_QUANTILES if '.quant' in variant else None,
        seed=42,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='.loggrowth.quant',
                    help='only checkpoints whose variant tag contains this')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(SAVE_DIR, 'MSAGAT-Net.*.pt')))
    done = skipped = 0
    for path in ckpts:
        m = TOKEN_RE.match(os.path.basename(path))
        if m is None or args.pattern not in (m.group('variant') or ''):
            continue
        if m.group('abl') != 'none':
            continue
        token = os.path.basename(path)[:-3]
        out = os.path.join(OUT_DIR, token + '.npy')
        if os.path.exists(out) and not args.force:
            skipped += 1
            continue

        ns = build_args(m.group('ds'), int(m.group('h')), m.group('variant') or '')
        state = torch.load(path, map_location='cpu')
        # The checkpoint dictates the quantile-head size (dev runs used 7
        # levels, the v2 campaign uses 23); the levels themselves do not
        # affect the attention matrix, only the head's parameter shapes.
        head_bias = state.get('quantile_head.proj.3.bias')
        if head_bias is not None and ns.quantiles is not None:
            n_levels = int(head_bias.shape[0]) + 1
            if n_levels != len(ns.quantiles):
                ns.quantiles = list(np.linspace(0.02, 0.98, n_levels))
        loader = DataBasicLoader(ns)
        model = MSAGATNet_Ablation(ns, loader)
        model.load_state_dict(state)
        if ns.cuda:
            model.cuda()
        model.eval()

        acc, count = None, 0
        with torch.no_grad():
            for X, Y, idx in loader.get_batches(loader.test, ns.batch, False):
                model(X, idx)
                a = model.graph_attention.attn        # [B, heads, N, N]
                a = a.mean(dim=(0, 1)).cpu().numpy()
                acc = a if acc is None else acc + a
                count += 1
        W = acc / count
        np.save(out, W)
        done += 1
        print(f'{token}: saved [{W.shape[0]}x{W.shape[1]}] '
              f'row-sum={W.sum(1).mean():.3f}')
    print(f'extracted {done}, skipped {skipped} (already present)')


if __name__ == '__main__':
    main()
