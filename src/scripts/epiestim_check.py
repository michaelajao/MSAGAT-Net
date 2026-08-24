"""Validate the renewal layer's learned R against a classical EpiEstim estimate.

The renewal decoder reads its backbone output as log R. That reading is only
meaningful if the quantity behaves like a reproduction number, so this compares
it against the Cori et al. (2013) estimator computed on the same series with a
fixed literature generation interval:

    R_t = sum_{s in window} I_s / sum_{s in window} Lambda_s
    Lambda_s = sum_tau w_tau I_{s-tau}

which is the posterior mean of the Cori estimator under a flat prior. The model
is never shown this quantity, so agreement is evidence rather than construction.

Reported per (dataset, horizon): Pearson and Spearman correlation over the
validation split, the fraction of time both sit on the same side of R = 1
(the epidemiologically meaningful threshold), and both medians.

Usage:
    python -m src.scripts.epiestim_check --pattern reniter
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE = os.path.join(BASE, 'save_attn')
RESULTS = os.path.join(BASE, 'report', 'results')
OUT_CSV = os.path.join(RESULTS, 'epiestim_validation.csv')

# Literature generation intervals. COVID: Ferretti et al. 2020 (mean 5.2 d,
# sd 1.72 d). Weekly ILI series cannot represent a sub-week kernel, so a
# 1-week-centred kernel is used and flagged as resolution-limited.
GI = {'nhs_timeseries': (5.2, 1.72), 'ltla_timeseries': (5.2, 1.72),
      'australia-covid': (5.2, 1.72), 'japan': (1.0, 0.6),
      'state360': (1.0, 0.6), 'region785': (1.0, 0.6)}
WEEKLY = {'japan', 'state360', 'region785'}


def discretised_gamma(mean, sd, L):
    lags = np.arange(1, L + 1, dtype=float)
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean
    w = lags ** (shape - 1) * np.exp(-lags / scale)
    return w / w.sum()


def epiestim_r(series, w, window=7):
    """Cori-style R_t over a trailing window. series: [T] incidence."""
    T, L = len(series), len(w)
    lam = np.zeros(T)
    for t in range(L, T):
        lam[t] = float(np.dot(w, series[t - L:t][::-1]))
    r = np.full(T, np.nan)
    for t in range(L + window, T):
        num = series[t - window + 1:t + 1].sum()
        den = lam[t - window + 1:t + 1].sum()
        r[t] = num / den if den > 1e-9 else np.nan
    return r


def model_r_nodes(ckpt, ns):
    """Per-node predicted R per validation timestep: [T_val, N]."""
    from ..data import DataBasicLoader
    from ..models import MSAGATNet_Ablation
    loader = DataBasicLoader(ns)
    model = MSAGATNet_Ablation(ns, loader)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    out = []
    with torch.no_grad():
        for X, Y, idx in loader.get_batches(loader.val, ns.batch, False):
            B, T, N = X.shape
            xt = X.permute(0, 2, 1).contiguous().view(B * N, 1, T)
            f = model.temp_conv(xt).view(B, N, -1)
            f = model.feature_process_low(f)
            f = model.feature_process_high(f)
            f = model.feature_act(model.feature_norm(f))
            gf, _ = model.graph_attention(f)
            fu = model.spatial_refinement_module(gf)
            pred = model.prediction_module(fu, X[:, -1, :]).transpose(1, 2)
            logR = pred.clamp(-model.renewal_logR_clamp, model.renewal_logR_clamp)
            out.append(torch.exp(logR).mean(dim=1).numpy())     # [B, N]
    return np.concatenate(out), loader


def model_r(ckpt, ns):
    """Mean predicted R per validation timestep, from the backbone's log R."""
    from ..data import DataBasicLoader
    from ..models import MSAGATNet_Ablation
    loader = DataBasicLoader(ns)
    model = MSAGATNet_Ablation(ns, loader)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    out = []
    with torch.no_grad():
        for X, Y, idx in loader.get_batches(loader.val, ns.batch, False):
            # replicate the backbone path, stopping at the log R output
            B, T, N = X.shape
            xt = X.permute(0, 2, 1).contiguous().view(B * N, 1, T)
            f = model.temp_conv(xt).view(B, N, -1)
            f = model.feature_process_low(f)
            f = model.feature_process_high(f)
            f = model.feature_act(model.feature_norm(f))
            gf, _ = model.graph_attention(f)
            fu = model.spatial_refinement_module(gf)
            pred = model.prediction_module(fu, X[:, -1, :]).transpose(1, 2)
            logR = pred.clamp(-model.renewal_logR_clamp, model.renewal_logR_clamp)
            out.append(torch.exp(logR).mean(dim=(1, 2)).numpy())   # [B]
    return np.concatenate(out), loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='reniter')
    ap.add_argument('--per-region', action='store_true')
    args = ap.parse_args()
    from ..train import DATASET_CONFIGS, DEFAULT_QUANTILES, TRAIN_DEFAULTS
    from argparse import Namespace

    rows = []
    pat = re.compile(r'MSAGAT-Net\.(?P<ds>[^.]+)\.w-20\.h-(?P<h>\d+)\.none'
                     r'\.seed-(?P<s>\d+)\..*\.pt$')
    seen = set()
    for f in sorted(glob.glob(os.path.join(SAVE, f'*{args.pattern}*.pt'))):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        ds, h = m.group('ds'), int(m.group('h'))
        # pin the LEARNED arm: frozen-kernel variants share the reniter token
        # and would otherwise win the sort order
        if 'giunif' in os.path.basename(f) or 'gifix' in os.path.basename(f):
            continue
        if (ds, h) in seen:
            continue
        seen.add((ds, h))
        cfg = DATASET_CONFIGS[ds]
        # rebuild with the SAME config the checkpoint was trained under,
        # derived from its filename (some carry renewal_gamma, some do not)
        tag = os.path.basename(f)
        em = re.search(r'\.exp-([A-Za-z0-9-]+)', tag)
        exp = em.group(1).replace('-', ',') if em else 'nodecay,regpre,reniter'
        lm = re.search(r'\.renewal(\d+)', tag)
        lag = int(lm.group(1)) if lm else 7
        ns = Namespace(
            dataset=ds, sim_mat=cfg['sim_mat'], window=20, horizon=h,
            train=0.6, val=0.2, test=0.2, batch=TRAIN_DEFAULTS['batch'],
            dropout=0.2, ablation='none', hidden_dim=32, attention_heads=4,
            attention_regularization_weight=1e-5, num_scales=4, kernel_size=3,
            feature_channels=16, bottleneck_dim=8, use_adj_prior=True,
            adj_weight=0.1, use_graph_bias=True, adaptive=False, gpu=0,
            cuda=False, save_dir=SAVE, mylog=False, highway_window=4,
            extra='', label='', pcc='', pprm_supervision='repeat',
            spatial_gate=False, target_space='loggrowth',
            quantiles=DEFAULT_QUANTILES, seed=42, attn_fix=False,
            attn_exp=exp, renewal=True, renewal_lag=lag, gi_fix=None)
        rm, loader = model_r(f, ns)

        raw = np.clip(loader.rawdat, 0, None).mean(axis=1)   # national mean
        gmean, gsd = GI[ds]
        w = discretised_gamma(gmean, gsd, 7)
        er = epiestim_r(raw, w)
        idx = np.array(list(loader.valid_set))
        er_v = er[idx][:len(rm)]
        rm_v = rm[:len(er_v)]
        ok = np.isfinite(er_v) & np.isfinite(rm_v)
        if ok.sum() < 10:
            continue
        pr = stats.pearsonr(rm_v[ok], er_v[ok])
        sp = stats.spearmanr(rm_v[ok], er_v[ok])
        agree = float(np.mean((rm_v[ok] > 1) == (er_v[ok] > 1)))
        rows.append((ds, h, pr[0], pr[1], sp[0], agree,
                     float(np.median(rm_v[ok])), float(np.median(er_v[ok]))))

        if args.per_region:
            rn, _ = model_r_nodes(f, ns)
            cors = []
            for j in range(loader.m):
                srs = np.clip(loader.rawdat[:, j], 0, None)
                erj = epiestim_r(srs, w)
                e = erj[idx][:rn.shape[0]]
                mj = rn[:len(e), j]
                okj = np.isfinite(e) & np.isfinite(mj)
                if okj.sum() >= 20 and np.std(e[okj]) > 1e-9 and np.std(mj[okj]) > 1e-9:
                    cors.append(stats.pearsonr(mj[okj], e[okj])[0])
            if cors:
                cors = np.array(cors)
                print(f"    per-region ({ds} h={h}): n={len(cors)} regions, "
                      f"median r={np.median(cors):.3f}, IQR "
                      f"[{np.percentile(cors,25):.3f}, {np.percentile(cors,75):.3f}], "
                      f"r>0.5 in {int((cors>0.5).sum())}/{len(cors)}")

    print(f"{'dataset':>16} {'h':>3} {'pearson':>9} {'p':>10} {'spearman':>9} "
          f"{'R>1 agree':>10} {'med R model':>12} {'med R epiestim':>15}")
    for r in rows:
        print(f"{r[0]:>16} {r[1]:>3} {r[2]:>9.3f} {r[3]:>10.2e} {r[4]:>9.3f} "
              f"{r[5]:>10.1%} {r[6]:>12.3f} {r[7]:>15.3f}")
    if rows:
        os.makedirs(RESULTS, exist_ok=True)
        import csv
        with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
            wri = csv.writer(fh)
            wri.writerow(['dataset', 'horizon', 'pearson', 'p_value',
                          'spearman', 'threshold_agreement',
                          'median_R_model', 'median_R_epiestim'])
            wri.writerows(rows)
        print(f"\nwrote {OUT_CSV}")
        print("NOTE weekly datasets (japan/state360/region785) use a "
              "1-week-centred kernel; a sub-week GI is unrepresentable there.")


if __name__ == '__main__':
    main()
