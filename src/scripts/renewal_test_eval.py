"""Test-split evaluation of the three renewal arms + the direct-decoder bar.

The single planned test contact for the renewal work (program.md hygiene: all
selection happened on validation; this reads the test predictions that the
training pipeline already persisted -- no model is re-run).

Statistics: per-cell mean +- sd over seeds, paired Wilcoxon across seeds
(learned vs each alternative), and per-timestep Diebold-Mariano (Newey-West
lag h-1, HLN correction) on the median-RMSE seed of each arm.
"""

import os
import sys

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED = os.path.join(BASE, 'report', 'predictions')
sys.path.insert(0, BASE)
from src.scripts.renewal_paper import token, SEEDS, CELLS, LAG  # noqa: E402


def bar_token(ds, h, seed):
    return (f'MSAGAT-Net.{ds}.w-20.h-{h}.none.seed-{seed}.with_adj.loggrowth'
            f'.quant.exp-nodecay-regpre')


def load(ds, h, seed, arm):
    t = bar_token(ds, h, seed) if arm == 'bar' else token(ds, h, seed, arm)
    p = os.path.join(PRED, ds, t + '.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d['y_true'], d['y_pred']


def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def dm_stat(e2a, e2b, h):
    d = e2a.mean(axis=1) - e2b.mean(axis=1)
    n = len(d); dbar = d.mean(); dc = d - dbar
    lag = max(h - 1, 0)
    lrv = np.mean(dc * dc)
    for k in range(1, min(lag, n - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * np.mean(dc[k:] * dc[:-k])
    lrv = max(lrv, 1e-300)
    dm = dbar / np.sqrt(lrv / n)
    dm *= np.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1e-12) / n)
    return dm, 2.0 * stats.t.sf(abs(dm), df=n - 1)


ARMS = ['bar', 'learned', 'uniform', 'fixed']
print(f"{'cell':>16} | " + " | ".join(f"{a:>16}" for a in ARMS))
print('-' * 92)
per = {}
for ds, h in CELLS:
    row = f"{ds[:10]+' h'+str(h):>16} |"
    for arm in ARMS:
        vals = []
        for s in SEEDS:
            d = load(ds, h, s, arm)
            if d is not None:
                vals.append(rmse(*d))
        per[(ds, h, arm)] = vals
        if vals:
            sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            row += f" {np.mean(vals):>9.3f}±{sd:5.3f} |"
        else:
            row += f" {'-':>16} |"
    print(row)

print("\npaired Wilcoxon across seeds (learned vs ...):")
for other in ['bar', 'uniform', 'fixed']:
    diffs, cells_won = [], 0
    for ds, h in CELLS:
        a, b = per[(ds, h, 'learned')], per[(ds, h, other)]
        n = min(len(a), len(b))
        if n >= 4:
            w = stats.wilcoxon(a[:n], b[:n])
            rel = (np.mean(a[:n]) - np.mean(b[:n])) / np.mean(b[:n]) * 100
            cells_won += int(rel < 0)
            diffs.append(rel)
            print(f"  {ds[:10]} h={h:<3} vs {other:>7}: {rel:+6.1f}%  p={w.pvalue:.3f}")
    if diffs:
        print(f"  -> vs {other}: better in {cells_won}/{len(diffs)} cells, "
              f"mean {np.mean(diffs):+.1f}%")

print("\nDM on median-RMSE seeds (learned vs bar):")
for ds, h in CELLS:
    def med_seed(arm):
        scored = sorted((rmse(*load(ds, h, s, arm)), s)
                        for s in SEEDS if load(ds, h, s, arm) is not None)
        return scored[len(scored) // 2][1]
    la = load(ds, h, med_seed('learned'), 'learned')
    ba = load(ds, h, med_seed('bar'), 'bar')
    n = min(la[0].shape[0], ba[0].shape[0])
    dm, p = dm_stat((la[0][-n:] - la[1][-n:]) ** 2,
                    (ba[0][-n:] - ba[1][-n:]) ** 2, h)
    print(f"  {ds[:10]} h={h:<3} dm={dm:+.3f} p={p:.4f} "
          f"({'learned better' if dm < 0 else 'bar better'})")
