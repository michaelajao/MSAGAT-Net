"""Measure what the spatial attention actually learned, from the parameters.

Finding E3 -- that EAGAM performs uniform mean aggregation rather than
attending -- is the load-bearing negative result of the evaluation paper,
and until now its numbers existed only in prose from a one-off measurement
of a few checkpoints.

This derives them from the state dicts directly, with no forward pass, so it
covers every checkpoint on disk rather than a sample, runs on CPU in
seconds, and does not compete with the GPU. It complements
`extract_attention.py`, which does run the model and persists the [N, N]
matrix; that script cannot rebuild the attention-revival checkpoints because
its `build_args` predates the `attn_exp` flag.

The argument does not need a forward pass. Attention logits are

    S = q k^T / sqrt(d)  +  U V  +  softplus(adj_scale) * A_normalised

and a softmax is selective only if the spread of S across a row is O(1). Two
of the three terms are computable from the checkpoint and the adjacency file
alone:

    uv_row_sd     within-row standard deviation of U @ V
    adj_row_sd    softplus(adj_scale) * within-row sd of the row-normalised
                  adjacency

If both are orders of magnitude below 1, the softmax is uniform whatever the
content term does, because the content term is bounded by the same
projections. Reported alongside:

    fusion_alpha_spread   max-min of softmax(fusion_weight), the MSSFM's
                          "locality-biased" hop weights -- zero means the
                          multi-hop mixing is a plain uniform average
    decay_at_h            exp(-exp(log_decay) * h), how much of PPRM's
                          persistence branch survives at the scored lead
    highway_sigma         sigmoid(highway_ratio), weight on the learned
                          forecast against the linear autoregression
    layernorm_gain_mean   should be 1.0; _init_weights overwrites every
                          LayerNorm gain with uniform(+/-1/sqrt(d)), so this
                          records the effect of that bug on trained models

    python -m src.scripts.attention_diagnostics

Writes `report/results/attention_diagnostics.csv`.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from src.tokens import parse_token  # noqa: E402

OUT = os.path.join(BASE, 'report', 'results', 'attention_diagnostics.csv')
SAVE_DIRS = ['save_all', 'save_attn', 'save_renewal']

DATASET_ADJ = {
    'japan': 'japan-adj', 'region785': 'region-adj', 'state360': 'state-adj-49',
    'australia-covid': 'australia-adj', 'ltla_timeseries': 'ltla-adj',
    'nhs_timeseries': 'nhs-adj',
}

_ADJ_CACHE = {}


def adjacency_row_sd(dataset, sim_mat=None):
    """Within-row sd of the row-normalised adjacency, as the model uses it."""
    name = sim_mat or DATASET_ADJ.get(dataset)
    if name is None:
        return np.nan
    if name not in _ADJ_CACHE:
        path = os.path.join(BASE, 'data', f'{name}.txt')
        if not os.path.exists(path):
            _ADJ_CACHE[name] = np.nan
        else:
            A = np.loadtxt(path, delimiter=',' if ',' in open(path).readline()
                           else None)
            A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
            _ADJ_CACHE[name] = float(np.mean(np.std(A, axis=1)))
    return _ADJ_CACHE[name]


def softplus(x):
    return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))


def diagnose(path):
    token = os.path.splitext(os.path.basename(path))[0]
    try:
        spec = parse_token(token)
    except ValueError:
        return None
    sd = torch.load(path, map_location='cpu', weights_only=True)
    get = lambda k: sd[k].float().numpy() if k in sd else None  # noqa: E731

    u, v = get('graph_attention.u'), get('graph_attention.v')
    rec = {
        'run_token': token,
        'dataset': spec['dataset'], 'horizon': spec['horizon'],
        'seed': spec['seed'], 'ablation': spec['ablation'],
        'target_space': spec['target_space'],
        'attn_exp': spec['attn_exp'] or '',
        'save_dir': os.path.basename(os.path.dirname(path)),
    }

    if u is not None and v is not None:
        uv = np.matmul(u, v)                       # [heads, N, N]
        rec['u_absmax'] = float(np.abs(u).max())
        rec['v_absmax'] = float(np.abs(v).max())
        rec['uv_row_sd'] = float(np.mean(np.std(uv, axis=-1)))
        rec['uv_absmax'] = float(np.abs(uv).max())
    else:
        rec.update(u_absmax=np.nan, v_absmax=np.nan,
                   uv_row_sd=np.nan, uv_absmax=np.nan)

    a = get('graph_attention.adj_scale')
    rec['adj_scale'] = float(a) if a is not None else np.nan
    rec['adj_scale_softplus'] = softplus(float(a)) if a is not None else np.nan
    rec['adj_row_sd_raw'] = adjacency_row_sd(spec['dataset'], spec['sim_mat'])
    rec['adj_logit_sd'] = rec['adj_scale_softplus'] * rec['adj_row_sd_raw']

    # Which term, if either, could make the softmax selective?
    rec['logit_sd_total'] = np.nansum([rec['uv_row_sd'], rec['adj_logit_sd']])
    rec['uv_share'] = (rec['uv_row_sd'] / rec['logit_sd_total']
                       if rec['logit_sd_total'] > 0 else np.nan)

    f = get('spatial_refinement_module.fusion_weight')
    if f is not None:
        e = np.exp(f - f.max())
        alpha = e / e.sum()
        rec['fusion_alpha_spread'] = float(alpha.max() - alpha.min())
        rec['fusion_alpha_max'] = float(alpha.max())
        rec['n_hops'] = int(len(alpha))
    else:
        rec.update(fusion_alpha_spread=np.nan, fusion_alpha_max=np.nan,
                   n_hops=np.nan)

    ld = get('prediction_module.log_decay')
    if ld is not None:
        gamma = float(np.exp(float(ld)))
        rec['pprm_gamma'] = gamma
        rec['pprm_decay_at_h'] = float(np.exp(-gamma * spec['horizon']))
    else:
        rec.update(pprm_gamma=np.nan, pprm_decay_at_h=np.nan)

    hr = get('highway_ratio')
    rec['highway_sigma'] = (float(1 / (1 + np.exp(-float(hr))))
                            if hr is not None else np.nan)

    lw = get('graph_attention.log_attention_reg_weight')
    rec['log_attn_reg_weight'] = float(lw) if lw is not None else np.nan

    gains = [sd[k].float().numpy() for k in sd
             if k.endswith('.weight') and sd[k].dim() == 1 and 'norm' in k.lower()]
    if gains:
        allg = np.concatenate([g.ravel() for g in gains])
        rec['layernorm_gain_mean'] = float(allg.mean())
        rec['layernorm_gain_absmean'] = float(np.abs(allg).mean())
        rec['layernorm_gain_negfrac'] = float((allg < 0).mean())
    else:
        rec.update(layernorm_gain_mean=np.nan, layernorm_gain_absmean=np.nan,
                   layernorm_gain_negfrac=np.nan)
    return rec


def main():
    rows = []
    for d in SAVE_DIRS:
        for path in sorted(glob.glob(os.path.join(BASE, d, 'MSAGAT-Net.*.pt'))):
            rec = diagnose(path)
            if rec:
                rows.append(rec)

    if not rows:
        print('no checkpoints found')
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option('display.width', 200)

    def arm(r):
        if r.attn_exp:
            return 'revived (' + r.attn_exp.split('-renew')[0] + ')'
        return 'v2 (loggrowth)' if r.target_space == 'loggrowth' else 'v1 (level)'

    df['arm'] = df.apply(arm, axis=1)
    core = df[df.arm.isin(['v1 (level)', 'v2 (loggrowth)',
                           'revived (nodecay-regpre)'])]

    print(f'{len(df)} checkpoints. Attention logit spread by arm '
          f'(a softmax is selective only when this is O(1)):\n')
    g = (core.groupby('arm')
             .agg(n=('run_token', 'size'),
                  u_absmax=('u_absmax', 'median'),
                  uv_row_sd=('uv_row_sd', 'median'),
                  adj_logit_sd=('adj_logit_sd', 'median'),
                  uv_share=('uv_share', 'median'),
                  fusion_spread=('fusion_alpha_spread', 'median'),
                  decay_at_h=('pprm_decay_at_h', 'median'),
                  highway=('highway_sigma', 'median')))
    print(g.to_string(float_format=lambda v: f'{v:.4g}'))

    print('\nBy dataset, frozen arms only (v1 and v2) -- u collapses everywhere:')
    fz = core[core.arm != 'revived (nodecay-regpre)']
    print(fz.groupby(['dataset', 'arm'])
            .agg(n=('run_token', 'size'), u_absmax=('u_absmax', 'median'),
                 uv_row_sd=('uv_row_sd', 'median'),
                 adj_logit_sd=('adj_logit_sd', 'median'))
            .to_string(float_format=lambda v: f'{v:.3g}'))

    print('\nLayerNorm gains (should be 1.0; _init_weights overwrites them):')
    print(core.groupby('arm')
              .agg(gain_mean=('layernorm_gain_mean', 'median'),
                   gain_absmean=('layernorm_gain_absmean', 'median'),
                   frac_negative=('layernorm_gain_negfrac', 'median'))
              .to_string(float_format=lambda v: f'{v:.3f}'))

    print(f'\nwrote {os.path.relpath(OUT, BASE)}')


if __name__ == '__main__':
    main()
