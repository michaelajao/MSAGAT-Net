"""Multi-seed confirmation of the renewal-layer interpretability results.

Every renewal finding so far is seed 42 on a five-cell proxy grid. The
attention campaign showed exactly how badly that can mislead: a result of 4/5
cells and -6.9% on the proxy became 8/21 cells and +2.1% under a 5-seed
full-grid confirmation. So none of the three claims below goes into a paper
until it survives multiple seeds.

Three arms on an identical backbone, so the kernel is the only thing varying:
    learned  - alpha free
    uniform  - alpha frozen flat (no generation-interval shape)
    fixed    - alpha frozen to gamma(5.2, 1.72), the Ferretti et al. 2020
               COVID generation interval

Run on the daily UK series, where a COVID generation interval is the
applicable literature value. Weekly ILI is deliberately excluded: a COVID GI
there is a 4.83-WEEK kernel, so any benefit is smoothing rather than mechanism.

Usage:
    python -m src.scripts.renewal_paper            # run everything
    python -m src.scripts.renewal_paper --report   # summarise what exists
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE = os.path.join(BASE, 'save_renewal')
PRED = os.path.join(BASE, 'report', 'predictions')

SEEDS = [42, 30, 45, 123, 1000]
CELLS = [('nhs_timeseries', 3), ('nhs_timeseries', 7), ('nhs_timeseries', 14)]
LAG = 7
GI_MEAN, GI_SD = 5.2, 1.72          # Ferretti et al. 2020, COVID
ARMS = {
    'learned': [],
    'uniform': ['giunif'],
    'fixed': None,                   # signalled via --gi_fix
}


def token(ds, h, seed, arm):
    exp = 'nodecay,regpre,reniter'
    if arm == 'uniform':
        exp += ',giunif'
    t = (f'MSAGAT-Net.{ds}.w-20.h-{h}.none.seed-{seed}.with_adj.loggrowth'
         f'.quant.exp-{exp.replace(",", "-")}.renewal{LAG}')
    if arm == 'fixed':
        t += f'.gifix{GI_MEAN:g}-{GI_SD:g}'
    return t


def run(ds, h, seed, arm):
    exp = 'nodecay,regpre,reniter' + (',giunif' if arm == 'uniform' else '')
    cmd = [sys.executable, '-m', 'src.train', '--single', '--dataset', ds,
           '--horizon', str(h), '--seed', str(seed),
           '--target_space', 'loggrowth', '--quantiles',
           '--attn_exp', exp, '--renewal', '--renewal_lag', str(LAG),
           '--save_dir', SAVE]
    if arm == 'fixed':
        cmd += ['--gi_fix', str(GI_MEAN), str(GI_SD)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=BASE, stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t0


def val_rmse(ds, h, seed, arm):
    p = os.path.join(PRED, ds, token(ds, h, seed, arm) + '.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return float(np.sqrt(np.mean((d['y_true_val'] - d['y_pred_val']) ** 2)))


def kernel(ds, h, seed, arm):
    p = os.path.join(SAVE, token(ds, h, seed, arm) + '.pt')
    if not os.path.exists(p):
        # seed-42 learned-arm checkpoints from the exploratory phase live in
        # save_attn under the same token
        p = os.path.join(BASE, 'save_attn', token(ds, h, seed, arm) + '.pt')
    if not os.path.exists(p):
        return None
    a = torch.softmax(torch.load(p, map_location='cpu')['log_alpha'], 0).numpy()
    return a


def report():
    print(f"{'cell':>18} | " + " | ".join(f"{a:>16}" for a in ARMS))
    print('-' * 78)
    acc = {a: [] for a in ARMS}
    for ds, h in CELLS:
        line = f"{ds[:12]+' h'+str(h):>18} |"
        for arm in ARMS:
            v = [val_rmse(ds, h, s, arm) for s in SEEDS]
            v = [x for x in v if x is not None]
            if not v:
                line += f" {'-':>16} |"
                continue
            m, sd = np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0.0)
            acc[arm].append((ds, h, m))
            line += f" {m:>9.3f}±{sd:5.3f} |"
        print(line)

    print("\nlearned vs each alternative, per cell:")
    for other in ('uniform', 'fixed'):
        d = []
        for (ds, h, ml) in acc['learned']:
            mo = [m for (d2, h2, m) in acc[other] if (d2, h2) == (ds, h)]
            if mo:
                d.append((ml - mo[0]) / mo[0] * 100)
        if d:
            print(f"  vs {other:>8}: learned better in {sum(1 for x in d if x < 0)}"
                  f"/{len(d)} cells, mean {np.mean(d):+.1f}%")

    print("\nrecovered generation interval (learned arm), mean delay in days:")
    for ds, h in CELLS:
        ks = [kernel(ds, h, s, 'learned') for s in SEEDS]
        ks = [k for k in ks if k is not None]
        if not ks:
            continue
        lags = np.arange(1, LAG + 1)
        md = [float((lags * k).sum()) for k in ks]
        inrange = sum(1 for x in md if 3.0 <= x <= 5.0)
        print(f"  {ds[:12]} h={h:<3} {np.mean(md):.2f} ± "
              f"{np.std(md, ddof=1) if len(md) > 1 else 0:.2f} d  "
              f"(n={len(md)} seeds; {inrange}/{len(md)} inside the published "
              f"3-5 d COVID range)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()
    if args.report:
        report()
        return
    os.makedirs(SAVE, exist_ok=True)
    todo = [(ds, h, s, a) for ds, h in CELLS for a in ARMS for s in SEEDS
            if val_rmse(ds, h, s, a) is None]
    print(f'{len(todo)} runs to do', flush=True)
    for i, (ds, h, s, a) in enumerate(todo, 1):
        rc, dt = run(ds, h, s, a)
        print(f'[{i}/{len(todo)}] {ds} h={h} seed={s} {a}: rc={rc} {dt:.0f}s',
              flush=True)
    report()


if __name__ == '__main__':
    main()
