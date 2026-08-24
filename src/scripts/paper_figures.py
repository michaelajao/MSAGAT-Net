"""Generate the figures for the PLOS renewal paper.

Print figures (PDF + 300 dpi PNG) into doc/plos-renewal/figs/. Every series is
recomputed from artifacts (checkpoints, prediction npz, raw data) so the
figures share provenance with the tables. Palette: validated categorical slots
(blue #2a78d6, orange #eb6834, aqua #1baf7a, yellow #eda100) with neutral gray
for the non-arm reference; every series is direct-labelled.
"""

import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(BASE, 'doc', 'plos-renewal', 'figs')
sys.path.insert(0, BASE)
from src.scripts.renewal_paper import kernel, SEEDS, CELLS, LAG  # noqa
from src.scripts.renewal_test_eval import load, rmse  # noqa
from src.scripts.epiestim_check import (discretised_gamma, epiestim_r, GI)  # noqa

os.makedirs(FIGS, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, GRAY = ('#2a78d6', '#eb6834', '#1baf7a',
                                    '#eda100', '#8a8a85')
plt.rcParams.update({
    'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'figure.dpi': 120, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
})


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGS, f'{name}.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f'wrote figs/{name}.pdf/.png')


# ---- F2: learned generation-interval kernels -------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.4), sharey=True)
lags = np.arange(1, LAG + 1)
for ax, (ds, h) in zip(axes, CELLS):
    ks = [kernel(ds, h, s, 'learned') for s in SEEDS]
    ks = np.array([k for k in ks if k is not None])
    ax.axvspan(3, 5, color=BLUE, alpha=0.10, lw=0)
    for k in ks:
        ax.plot(lags, k, color=GRAY, lw=0.8, alpha=0.6)
    mean_k = ks.mean(axis=0)
    ax.plot(lags, mean_k, color=BLUE, lw=2.0)
    md = float((lags * mean_k).sum())
    ax.set_title(f'$h={h}$', fontsize=9)
    ax.annotate(f'mean delay {md:.2f} d', xy=(0.97, 0.92),
                xycoords='axes fraction', ha='right', fontsize=8,
                color='#404040')
    ax.set_xlabel('delay from forecast step (days)')
    ax.set_xticks(lags)
axes[0].set_ylabel(r'learned $\alpha_\tau$')
axes[0].annotate('published COVID GI\n(3–5 d)', xy=(3.1, 0.02), fontsize=7,
                 color=BLUE, alpha=0.9)
axes[0].plot([], [], color=GRAY, lw=0.8, label='individual seeds (n=5)')
axes[0].plot([], [], color=BLUE, lw=2.0, label='seed mean')
axes[0].legend(frameon=False, fontsize=7, loc='upper left',
               bbox_to_anchor=(0.0, 0.88))
save(fig, 'F2_gi_kernels')

# ---- F3: model R vs EpiEstim (NHS h=3, learned arm) ------------------------
from src.train import DATASET_CONFIGS, DEFAULT_QUANTILES, TRAIN_DEFAULTS  # noqa
from src.scripts.epiestim_check import model_r  # noqa
from argparse import Namespace

ds, h = 'nhs_timeseries', 3
ckpts = [f for f in glob.glob(os.path.join(BASE, 'save_attn', '*reniter*.pt'))
         if 'renewres' not in f]
ckpts = [f for f in ckpts
         if 'giunif' not in f and 'gifix' not in f
         and f'.{ds}.' in f.replace('MSAGAT-Net.', '.')
         or (os.path.basename(f).startswith(f'MSAGAT-Net.{ds}.w-20.h-{h}.')
             and 'giunif' not in f and 'gifix' not in f and 'reniter' in f)]
# first in sorted order == the checkpoint epiestim_check.py analyses,
# so the figure and Table 3 share one artifact
ckpt = sorted(c for c in ckpts
              if f'h-{h}.' in os.path.basename(c) and f'{ds}' in c)[0]
cfg = DATASET_CONFIGS[ds]
ns = Namespace(dataset=ds, sim_mat=cfg['sim_mat'], window=20, horizon=h,
               train=0.6, val=0.2, test=0.2, batch=TRAIN_DEFAULTS['batch'],
               dropout=0.2, ablation='none', hidden_dim=32, attention_heads=4,
               attention_regularization_weight=1e-5, num_scales=4,
               kernel_size=3, feature_channels=16, bottleneck_dim=8,
               use_adj_prior=True, adj_weight=0.1, use_graph_bias=True,
               adaptive=False, gpu=0, cuda=False,
               save_dir=os.path.join(BASE, 'save_attn'), mylog=False,
               highway_window=4, extra='', label='', pcc='',
               pprm_supervision='repeat', spatial_gate=False,
               target_space='loggrowth', quantiles=DEFAULT_QUANTILES, seed=42,
               attn_fix=False, attn_exp='nodecay,regpre,reniter',
               renewal=True, renewal_lag=7, gi_fix=None)
rm, loader = model_r(ckpt, ns)
raw = np.clip(loader.rawdat, 0, None).mean(axis=1)
w = discretised_gamma(*GI[ds], 7)
er = epiestim_r(raw, w)
idx = np.array(list(loader.valid_set))
er_v = er[idx][:len(rm)]
rm_v = rm[:len(er_v)]
ok = np.isfinite(er_v) & np.isfinite(rm_v)
r_pearson = float(np.corrcoef(rm_v[ok], er_v[ok])[0, 1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6),
                               gridspec_kw={'width_ratios': [2, 1]})
t = np.arange(len(rm_v))
ax1.axhline(1.0, color='#b0b0ab', lw=0.8, ls='--')
ax1.plot(t[ok], er_v[ok], color=ORANGE, lw=1.6)
ax1.plot(t[ok], rm_v[ok], color=BLUE, lw=1.6)
ax1.annotate('EpiEstim', xy=(t[ok][-1], er_v[ok][-1]), xytext=(4, 0),
             textcoords='offset points', color=ORANGE, fontsize=8, va='center')
ax1.annotate('model', xy=(t[ok][-1], rm_v[ok][-1]), xytext=(4, -8),
             textcoords='offset points', color=BLUE, fontsize=8, va='center')
ax1.annotate('$R=1$', xy=(1, 1.0), xytext=(0, 3), textcoords='offset points',
             fontsize=7, color='#707070')
ax1.set_xlabel('validation day (NHS, $h=3$)')
ax1.set_ylabel('reproduction number $R$')
ax1.set_xlim(0, t[ok][-1] + 14)
ax2.plot([0.5, 1.6], [0.5, 1.6], color='#b0b0ab', lw=0.8, ls='--')
ax2.scatter(er_v[ok], rm_v[ok], s=10, color=BLUE, alpha=0.55,
            edgecolors='white', linewidths=0.4)
ax2.annotate(f'$r = {r_pearson:.2f}$', xy=(0.05, 0.9),
             xycoords='axes fraction', fontsize=9)
ax2.set_xlabel('EpiEstim $R$')
ax2.set_ylabel('model $R$')
save(fig, 'F3_r_validation')

# ---- F4: test-split accuracy, four arms ------------------------------------
ARMS = [('bar', 'direct', GRAY), ('learned', 'learned', BLUE),
        ('uniform', 'uniform', AQUA), ('fixed', 'fixed (lit.)', YELLOW)]
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.4))
for ax, (ds, h) in zip(axes, CELLS):
    for i, (arm, lbl, col) in enumerate(ARMS):
        vals = [rmse(*load(ds, h, s, arm)) for s in SEEDS
                if load(ds, h, s, arm) is not None]
        ax.bar(i, np.mean(vals), width=0.62, color=col, alpha=0.30, zorder=2)
        ax.scatter([i + np.random.default_rng(7 + i).uniform(-0.10, 0.10)
                    for _ in vals], vals, s=14, color=col, zorder=3,
                   edgecolors='white', linewidths=0.5)
        ax.plot([i - 0.31, i + 0.31], [np.mean(vals)] * 2, color=col, lw=2,
                zorder=4)
    ax.set_title(f'$h={h}$', fontsize=9)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([l for _, l, _ in ARMS], rotation=28, ha='right',
                       fontsize=7)
axes[0].set_ylabel('test RMSE (NHS)')
save(fig, 'F4_arms_test')

# ---- F5: the horizon threshold ---------------------------------------------
# (a) attention fix, test-split change vs frozen v2, daily datasets mean
pat = re.compile(r'MSAGAT-Net\.(?P<ds>[^.]+)\.w-20\.h-(?P<h>\d+)\.none'
                 r'\.seed-(?P<s>\d+)\.with_adj\.loggrowth\.quant'
                 r'(?P<fix>\.exp-nodecay-regpre)?\.npz$')
vals = {}
for f in glob.glob(os.path.join(BASE, 'report', 'predictions', '*',
                                'MSAGAT-Net*.npz')):
    m = pat.match(os.path.basename(f))
    if not m or m.group('ds') not in ('nhs_timeseries', 'ltla_timeseries',
                                      'australia-covid'):
        continue
    d = np.load(f)
    key = (m.group('ds'), int(m.group('h')),
           'fix' if m.group('fix') else 'v2')
    vals.setdefault(key, {})[int(m.group('s'))] = float(
        np.sqrt(np.mean((d['y_true'] - d['y_pred']) ** 2)))
att = {}
for hh in (3, 7, 14):
    ds_deltas = []
    for dds in ('nhs_timeseries', 'ltla_timeseries', 'australia-covid'):
        a, b = vals.get((dds, hh, 'fix'), {}), vals.get((dds, hh, 'v2'), {})
        com = sorted(set(a) & set(b))
        if len(com) >= 5:
            fa = np.mean([a[s] for s in com]); fb = np.mean([b[s] for s in com])
            ds_deltas.append((fa - fb) / fb * 100)
    att[hh] = float(np.mean(ds_deltas))

# (b) learned residual weight gamma, NHS, broken vs iterated formulation
def gammas(pattern):
    out = {}
    for f in glob.glob(os.path.join(BASE, 'save_attn', pattern)):
        b = os.path.basename(f)
        if 'nhs_timeseries' not in b:
            continue
        hh = int(re.search(r'\.h-(\d+)\.', b).group(1))
        g = torch.load(f, map_location='cpu').get('renewal_gamma')
        if g is not None:
            out.setdefault(hh, []).append(float(g))
    return {k: float(np.mean(v)) for k, v in out.items()}

g_broken = gammas('*regpre-renewres.renewal*.pt')
g_iter = gammas('*reniter-renewres.renewal*.pt')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
hs = [3, 7, 14]
cols = [ORANGE if att[h_] > 0 else BLUE for h_ in hs]
ax1.axhline(0, color='#707070', lw=0.8)
ax1.bar(range(3), [att[h_] for h_ in hs], width=0.55, color=cols, alpha=0.85)
lo = min(min(att.values()), 0); hi = max(max(att.values()), 0)
ax1.set_ylim(lo - 2.8, hi + 2.8)
for i, h_ in enumerate(hs):
    va = 'bottom' if att[h_] > 0 else 'top'
    off = 0.4 if att[h_] > 0 else -0.4
    ax1.annotate(f'{att[h_]:+.1f}%', xy=(i, att[h_] + off), ha='center',
                 va=va, fontsize=8)
ax1.set_xticks(range(3)); ax1.set_xticklabels([f'$h={h_}$' for h_ in hs])
ax1.set_ylabel(r'$\Delta$ test RMSE, attention fix (%)')
ax1.set_title('(a) working spatial attention', fontsize=9)
ax2.axhline(0, color='#707070', lw=0.8)
ax2.plot(hs, [g_broken.get(h_, np.nan) for h_ in hs], color=ORANGE, lw=1.8,
         marker='o', ms=5, ls='--')
ax2.plot(hs, [g_iter.get(h_, np.nan) for h_ in hs], color=BLUE, lw=1.8,
         marker='o', ms=5)
ax2.annotate('stale kernel', xy=(hs[-1], g_broken.get(14, 0)), xytext=(5, 0),
             textcoords='offset points', color=ORANGE, fontsize=8, va='center')
ax2.annotate('iterated', xy=(hs[-1], g_iter.get(14, 0)), xytext=(5, 0),
             textcoords='offset points', color=BLUE, fontsize=8, va='center')
ax2.set_xticks(hs); ax2.set_xticklabels([f'$h={h_}$' for h_ in hs])
ax2.set_ylabel(r'learned renewal weight $\gamma$')
ax2.set_title('(b) how much renewal the model keeps', fontsize=9)
fig.tight_layout()
save(fig, 'F5_horizon_threshold')
print('done')
