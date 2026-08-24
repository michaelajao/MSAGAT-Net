"""Single-step baseline retraining campaign across the colagnn and EpiGNN repos.

Retrains all five paper baselines under the corrected evaluation protocol
(single lead-h target, full-length test set, splits identical to MSAGAT-Net)
with 5 seeds per configuration. Each run is a subprocess with the working
directory set to the owning repo; per-timestep test predictions land in
MSAGAT-Net/report/predictions/ via the trainers' own dump code.

Usage (any cwd):
    python -m src.scripts.baseline_campaign            # run everything
    python -m src.scripts.baseline_campaign --dry-run
    python -m src.scripts.baseline_campaign --models cola_gnn epignn
"""

import argparse
import os
import subprocess
import sys
import threading
import time

GITHUB = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
COLAGNN = os.path.join(GITHUB, 'colagnn')
EPIGNN = os.path.join(GITHUB, 'EpiGNN')
PRED_DIR = os.path.join(GITHUB, 'MSAGAT-Net', 'report', 'predictions')
LOG_DIR = os.path.join(GITHUB, 'MSAGAT-Net', 'report', 'logs')

SEEDS = [42, 30, 45, 123, 1000]

# Ordered so the headline UK datasets produce numbers first.
DATASETS = [
    ('ltla_timeseries', 'ltla-adj', [3, 7, 14]),
    ('nhs_timeseries', 'nhs-adj', [3, 7, 14]),
    ('japan', 'japan-adj', [3, 5, 10, 15]),
    ('state360', 'state-adj-49', [3, 5, 10, 15]),
    ('region785', 'region-adj', [3, 5, 10, 15]),
    ('australia-covid', 'australia-adj', [3, 7, 14]),
]

COLAGNN_MODELS = ['cola_gnn', 'CNNRNN_Res', 'lstnet', 'dcrnn']
ALL_MODELS = COLAGNN_MODELS + ['epignn']

# Growth-space generality experiment: do log-growth targets lift the
# baselines too? Two representative baselines (one temporal, one graph)
# on the datasets where growth space mattered most for MSAGAT-Net.
# cola_gnn x LTLA is excluded (hours per run, LSTNet covers LTLA).
GROWTH_SPECS = {
    'lstnet': ['nhs_timeseries', 'region785', 'australia-covid',
               'ltla_timeseries'],
    'cola_gnn': ['nhs_timeseries', 'region785', 'australia-covid'],
}


def npz_path(model, dataset, horizon, seed, growth=False):
    tag = model + ('_lg' if growth else '')
    return os.path.join(PRED_DIR, dataset,
                        f'{tag}.{dataset}.w-20.h-{horizon}.none.seed-{seed}.npz')


def build_cmd(model, dataset, sim_mat, horizon, seed, growth=False):
    if model == 'epignn':
        return EPIGNN, [sys.executable, os.path.join('src', 'train.py'),
                        '--dataset', dataset, '--sim_mat', sim_mat,
                        '--horizon', str(horizon), '--seed', str(seed),
                        '--gpu', '0', '--cuda',
                        '--train', '.6', '--val', '.2', '--test', '.2']
    cmd = [sys.executable, os.path.join('src', 'train.py'),
           '--model', model, '--dataset', dataset,
           '--sim_mat', sim_mat, '--horizon', str(horizon),
           '--seed', str(seed), '--gpu', '0']
    if growth:
        cmd += ['--target_space', 'loggrowth']
    return COLAGNN, cmd


def is_heavy(model, dataset):
    """cola_gnn's O(N^2) attention and dcrnn's per-step seq2seq decoding make
    LTLA (372 nodes) runs take hours; schedule those last so every fast cell
    finishes first."""
    return model in ('cola_gnn', 'dcrnn') and dataset == 'ltla_timeseries'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--models', nargs='+', default=ALL_MODELS,
                    choices=ALL_MODELS)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--parallel', type=int, default=1,
                    help='concurrent training subprocesses (small models '
                         'underutilize the GPU; 2-3 is safe)')
    ap.add_argument('--growth', action='store_true',
                    help='run the growth-space generality experiment '
                         '(GROWTH_SPECS models/datasets with loggrowth targets)')
    args = ap.parse_args()

    if args.growth:
        specs = [(m, d, adj, h, s, True)
                 for d, adj, horizons in DATASETS
                 for m, allowed in GROWTH_SPECS.items() if d in allowed
                 for h in horizons
                 for s in SEEDS]
    else:
        specs = [(m, d, adj, h, s, False)
                 for d, adj, horizons in DATASETS
                 for h in horizons
                 for m in args.models
                 for s in SEEDS]
    specs.sort(key=lambda t: is_heavy(t[0], t[1]))   # heavy cells last
    todo = [s for s in specs
            if args.force or not os.path.exists(
                npz_path(s[0], s[1], s[3], s[4], growth=s[5]))]
    print(f'{len(specs)} runs total, {len(specs) - len(todo)} already done, '
          f'{len(todo)} to run (parallel={args.parallel})', flush=True)
    if args.dry_run:
        for m, d, _, h, s, growth in todo[:40]:
            print(f'  {m}{"_lg" if growth else ""}.{d}.h-{h}.seed-{s}')
        if len(todo) > 40:
            print(f'  ... and {len(todo) - 40} more')
        return 0

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, 'baseline_campaign.log')
    lock = threading.Lock()
    state = {'done': 0, 'failed': 0}

    run_log_dir = os.path.join(LOG_DIR, 'runs')
    os.makedirs(run_log_dir, exist_ok=True)

    def run_one(spec):
        m, d, adj, h, s, growth = spec
        cwd, cmd = build_cmd(m, d, adj, h, s, growth=growth)
        tag = m + ('_lg' if growth else '')
        # Per-run stdout+stderr goes to its own file: a discarded traceback
        # costs hours of re-running to diagnose on the heavy LTLA cells.
        run_log = os.path.join(run_log_dir, f'{tag}.{d}.h-{h}.seed-{s}.log')
        t0 = time.time()
        with open(run_log, 'w', encoding='utf-8') as out:
            result = subprocess.run(cmd, cwd=cwd, stdout=out,
                                    stderr=subprocess.STDOUT)
        ok = (result.returncode == 0
              and os.path.exists(npz_path(m, d, h, s, growth=growth)))
        with lock:
            state['done'] += 1
            if not ok:
                state['failed'] += 1
            print(f'[{state["done"]}/{len(todo)}] {m}.{d}.h-{h}.seed-{s} '
                  f'{"ok" if ok else "FAIL"} {time.time() - t0:.0f}s', flush=True)
            with open(log_path, 'a', encoding='utf-8') as fh:
                detail = '' if ok else f' log={os.path.basename(run_log)}'
                fh.write(f'{time.strftime("%Y%m%d_%H%M%S")} {tag}.{d}.h-{h}.seed-{s} '
                         f'{"ok" if ok else f"FAIL rc={result.returncode}"} '
                         f'{time.time() - t0:.0f}s{detail}\n')

    if args.parallel <= 1:
        for spec in todo:
            run_one(spec)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            list(pool.map(run_one, todo))

    print(f'baseline campaign finished: {state["done"] - state["failed"]}'
          f'/{len(todo)} ok, {state["failed"]} failed', flush=True)
    return 1 if state['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
