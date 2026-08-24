"""Attention-revival autoresearch harness (see program.md).

Runs one experiment configuration on the proxy grid, computes the acceptance
gate from the trained checkpoint, and appends everything to the campaign
ledger. Ratchet metric: validation RMSE (original units), averaged over proxy
cells. The test split is never read here.

Proxy grid (chosen for cycle time; documented per program.md):
    nhs_timeseries  h in {3, 7, 14}   (7 nodes,  daily,  ~2 min/run)
    japan           h in {3, 5}       (47 nodes, weekly, ~3 min/run)
seed 42 only -- survivors get the full 5-seed treatment at campaign end.

Gate (program.md): a run is discarded unless
    1. mean normalised attention row entropy <= 0.98, and
    2. content + U@V share of pre-softmax score variance >= 0.10,
       computed as var(content+bias) / (var(content+bias) + var(adj term))
       per row, averaged over the validation forward.

Usage:
    python -m src.scripts.attn_revival --exp nodecay,regpre
    python -m src.scripts.attn_revival --baseline noagam   # trains the bar
    python -m src.scripts.attn_revival --baseline v2       # from existing npz
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED = os.path.join(BASE, 'report', 'predictions')
SAVE = os.path.join(BASE, 'save_attn')
RESULTS = os.path.join(BASE, 'report', 'results')
RUNS_CSV = os.path.join(RESULTS, 'attn_revival_runs.csv')
LEDGER_CSV = os.path.join(RESULTS, 'attn_revival_ledger.csv')

PROXY = [('nhs_timeseries', 3), ('nhs_timeseries', 7), ('nhs_timeseries', 14),
         ('japan', 3), ('japan', 5)]
SEED = 42

# set by --renewal / --renewal-lag; consumed when rebuilding for the gate
RENEWAL = {'on': False, 'lag': 0, 'gi': None}


def token(ds, h, ablation='none', exp=''):
    t = f'MSAGAT-Net.{ds}.w-20.h-{h}.{ablation}.seed-{SEED}.with_adj.loggrowth.quant'
    if exp:
        t += '.exp-' + exp.replace(',', '-')
    if RENEWAL['on']:
        t += f".renewal{RENEWAL['lag'] or ''}"
    if RENEWAL['gi']:
        t += f".gifix{RENEWAL['gi'][0]:g}-{RENEWAL['gi'][1]:g}"
    return t


def val_rmse_from_npz(path):
    d = np.load(path)
    return float(np.sqrt(np.mean((d['y_true_val'] - d['y_pred_val']) ** 2)))


def gate_quantities(ds, h, exp):
    """Forward the val split through the checkpoint; return gate metrics."""
    from ..data import DataBasicLoader
    from ..models import MSAGATNet_Ablation
    from ..train import DATASET_CONFIGS, DEFAULT_QUANTILES, TRAIN_DEFAULTS
    from argparse import Namespace

    cfg = DATASET_CONFIGS[ds]
    ns = Namespace(
        dataset=ds, sim_mat=cfg['sim_mat'], window=TRAIN_DEFAULTS['window'],
        horizon=h, train=0.6, val=0.2, test=0.2,
        batch=TRAIN_DEFAULTS['batch'], dropout=TRAIN_DEFAULTS['dropout'],
        ablation='none', hidden_dim=TRAIN_DEFAULTS['hidden_dim'],
        attention_heads=TRAIN_DEFAULTS['attention_heads'],
        attention_regularization_weight=1e-5,
        num_scales=TRAIN_DEFAULTS['num_scales'], kernel_size=3,
        feature_channels=16, bottleneck_dim=TRAIN_DEFAULTS['bottleneck_dim'],
        use_adj_prior=True, adj_weight=0.1, use_graph_bias=True,
        adaptive=False, gpu=0, cuda=False, save_dir=SAVE, mylog=False,
        highway_window=4, extra='', label='', pcc='',
        pprm_supervision='repeat', spatial_gate=False,
        target_space='loggrowth', quantiles=DEFAULT_QUANTILES, seed=SEED,
        attn_fix=False, attn_exp=exp,
        renewal=RENEWAL['on'], renewal_lag=RENEWAL['lag'],
        gi_fix=RENEWAL['gi'],
    )
    ckpt = os.path.join(SAVE, token(ds, h, exp=exp) + '.pt')
    loader = DataBasicLoader(ns)
    model = MSAGATNet_Ablation(ns, loader)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    sa = model.graph_attention
    # (renewal gate quantities are read from `model` below)

    ents, shares = [], []
    feats = {}

    def grab(mod, inp, out):
        feats['in'] = inp[0]

    hook = sa.register_forward_hook(grab)
    with torch.no_grad():
        for X, Y, idx in loader.get_batches(loader.val, ns.batch, False):
            model(X, idx)
            A = sa.attn
            n = A.shape[-1]
            ent = -(A * torch.log(A.clamp_min(1e-12))).sum(-1) / math.log(n)
            ents.append(ent.reshape(-1).numpy())
            x = feats['in']
            B_, N_, H_ = x.shape
            qkv = sa.qkv_proj_high(sa.qkv_proj_low(x)).chunk(3, dim=-1)
            q, k, _ = [t.view(B_, N_, sa.heads, sa.head_dim).transpose(1, 2)
                       for t in qkv]
            content = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(sa.head_dim)
            bias = torch.matmul(sa.u, sa.v).unsqueeze(0)
            learned = content + bias
            adj_term = torch.zeros_like(learned)
            if sa.adj_prior is not None:
                adj_term = (F.softplus(sa.adj_scale)
                            * sa.adj_prior.unsqueeze(0).unsqueeze(0)
                            ).expand_as(learned)
            var_l = learned.var(dim=-1)
            var_a = adj_term.var(dim=-1)
            shares.append((var_l / (var_l + var_a + 1e-30)).reshape(-1).numpy())
    hook.remove()
    ents = np.concatenate(ents)
    shares = np.concatenate(shares)
    temp = (float(torch.exp(sa.log_attn_temp))
            if sa.log_attn_temp is not None else 1.0)
    extra = {}
    if getattr(model, 'renewal', False):
        # Renewal gate (program.md): the kernel must be a genuine delay
        # distribution, not a near-delta on one lag.
        a = torch.softmax(model.log_alpha.detach(), 0).numpy()
        lags = np.arange(len(a)) + (0 if model.renewal_from_zero else 1)
        ent = float(-(a * np.log(np.maximum(a, 1e-12))).sum() / math.log(len(a)))
        g = getattr(model, 'renewal_gamma', None)
        extra = {'gamma': (float(g) if g is not None else 1.0),
                 'alpha_first': float(a[0]),
                 'alpha_mean_lag': float((lags * a).sum()),
                 'alpha_entropy': ent,
                 'alpha': ' '.join(f'{x:.4f}' for x in a)}
    return {
        'ent_mean': float(ents.mean()), 'ent_min': float(ents.min()),
        'uv_share': float(shares.mean()),
        'u_absmax': float(sa.u.abs().max()), 'v_absmax': float(sa.v.abs().max()),
        'temp': temp, **extra,
    }


def append_row(path, row):
    """Append a row, widening the header if the row introduces new columns.

    Two hazards this handles, both hit in practice:
      * Taking fieldnames from each row while writing a header only at file
        creation silently MISALIGNS every row written after a new key appears.
      * Legacy rows may hold more values than the header, in which case
        DictReader buckets the surplus under a None key, which is not
        writable. Those surplus values are unrecoverable (their column names
        were never recorded), so they are dropped rather than guessed at.
    """
    if not os.path.exists(path):
        with open(path, 'a', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
        return

    with open(path, newline='', encoding='utf-8') as fh:
        rd = csv.reader(fh)
        rows = list(rd)
    header = rows[0] if rows else []
    body = rows[1:] if len(rows) > 1 else []

    new_cols = [c for c in row if c not in header]
    if not new_cols:
        with open(path, 'a', newline='', encoding='utf-8') as fh:
            csv.DictWriter(fh, fieldnames=header).writerow(row)
        return

    header = header + new_cols
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in body:
            w.writerow(r[:len(header)] + [''] * max(0, len(header) - len(r)))
        w.writerow([row.get(c, '') for c in header])


def run_cell(ds, h, exp='', ablation='none'):
    cmd = [sys.executable, '-m', 'src.train', '--single',
           '--dataset', ds, '--horizon', str(h), '--seed', str(SEED),
           '--ablation', ablation, '--target_space', 'loggrowth',
           '--quantiles', '--save_dir', SAVE]
    if exp:
        cmd += ['--attn_exp', exp]
    if RENEWAL['on']:
        cmd += ['--renewal']
        if RENEWAL['lag']:
            cmd += ['--renewal_lag', str(RENEWAL['lag'])]
        if RENEWAL['gi']:
            cmd += ['--gi_fix', str(RENEWAL['gi'][0]), str(RENEWAL['gi'][1])]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=BASE, stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default=None,
                    help='experiment token string, e.g. nodecay,regpre')
    ap.add_argument('--baseline', choices=['noagam', 'v2'], default=None)
    ap.add_argument('--note', default='')
    ap.add_argument('--renewal', action='store_true')
    ap.add_argument('--renewal-lag', type=int, default=0)
    ap.add_argument('--gi-fix', nargs=2, type=float, default=None,
                    metavar=('MEAN', 'SD'))
    args = ap.parse_args()
    RENEWAL['on'] = args.renewal
    RENEWAL['lag'] = args.renewal_lag
    RENEWAL['gi'] = tuple(args.gi_fix) if args.gi_fix else None
    os.makedirs(SAVE, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')

    if args.baseline == 'v2':
        # frozen v2 reference: read validation predictions already on disk
        rmses = {}
        for ds, h in PROXY:
            npz = os.path.join(PRED, ds, token(ds, h) + '.npz')
            rmses[f'{ds}.h{h}'] = val_rmse_from_npz(npz)
        mean = float(np.mean(list(rmses.values())))
        append_row(LEDGER_CSV, {
            'stamp': stamp, 'kind': 'baseline-v2', 'exp': '',
            'mean_val_rmse': round(mean, 4),
            **{k: round(v, 4) for k, v in rmses.items()},
            'ent_mean': '', 'ent_min': '', 'uv_share': '',
            'gate_pass': '', 'note': 'frozen v2, from existing npz'})
        print(f'baseline-v2 mean val RMSE {mean:.4f}  ' + str(rmses))
        return

    exp = '' if args.baseline == 'noagam' else args.exp
    ablation = 'no_agam' if args.baseline == 'noagam' else 'none'
    kind = 'baseline-noagam' if args.baseline == 'noagam' else 'config'
    if kind == 'config' and not exp:
        sys.exit('need --exp or --baseline')

    rmses, gates = {}, []
    for ds, h in PROXY:
        npz = os.path.join(PRED, ds, token(ds, h, ablation, exp) + '.npz')
        if not os.path.exists(npz):
            rc, dt = run_cell(ds, h, exp, ablation)
            print(f'{ds} h={h}: rc={rc} {dt:.0f}s', flush=True)
        rmses[f'{ds}.h{h}'] = val_rmse_from_npz(npz)
        row = {'stamp': stamp, 'kind': kind, 'exp': exp,
               'dataset': ds, 'horizon': h,
               'val_rmse': round(rmses[f'{ds}.h{h}'], 4)}
        if kind == 'config':
            g = gate_quantities(ds, h, exp)
            gates.append(g)
            row.update({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in g.items()})
        append_row(RUNS_CSV, row)

    mean = float(np.mean(list(rmses.values())))
    summary = {'stamp': stamp, 'kind': kind, 'exp': exp,
               'mean_val_rmse': round(mean, 4),
               **{k: round(v, 4) for k, v in rmses.items()}}
    if kind == 'config':
        ent_mean = float(np.mean([x['ent_mean'] for x in gates]))
        ent_min = float(np.min([x['ent_min'] for x in gates]))
        uv = float(np.mean([x['uv_share'] for x in gates]))
        gate_ok = bool(ent_mean <= 0.98 and uv >= 0.10)
        summary.update({'ent_mean': round(ent_mean, 4),
                        'ent_min': round(ent_min, 4),
                        'uv_share': round(uv, 4)})
        if RENEWAL['on'] and 'alpha_first' in gates[0]:
            af = float(np.mean([x['alpha_first'] for x in gates]))
            ml = float(np.mean([x['alpha_mean_lag'] for x in gates]))
            ae = float(np.mean([x['alpha_entropy'] for x in gates]))
            summary.update({'alpha_first': round(af, 4),
                            'alpha_mean_lag': round(ml, 3),
                            'alpha_entropy': round(ae, 4)})
            gate_ok = gate_ok and af <= 0.60 and ml >= 1.5 and ae >= 0.50
        summary['gate_pass'] = gate_ok
    else:
        summary.update({'ent_mean': '', 'ent_min': '', 'uv_share': '',
                        'gate_pass': ''})
    summary['note'] = args.note
    append_row(LEDGER_CSV, summary)
    print(f'{kind} exp="{exp}" mean val RMSE {mean:.4f}')
    if kind == 'config':
        print(f'gate: ent_mean={summary["ent_mean"]} '
              f'uv_share={summary["uv_share"]} pass={summary["gate_pass"]}')


if __name__ == '__main__':
    main()
