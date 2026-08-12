"""Campaign driver for the resubmission experiment matrix.

Enumerates (dataset, horizon, ablation, seed, sim_mat, variant) tuples per named
chunk and executes each as a fresh subprocess of ``python -m src.train --single``.
Fresh processes avoid CUDA memory fragmentation across hundreds of runs; the
skip-existing check makes any chunk resumable after an interruption.

Usage (from the repository root, conda env dl_env):
    python -m src.scripts.campaign --chunk main_uk
    python -m src.scripts.campaign --chunk devloop
    python -m src.scripts.campaign --chunk ablation --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED_DIR = os.path.join(BASE_DIR, 'report', 'predictions')
LOG_DIR = os.path.join(BASE_DIR, 'report', 'logs')

SEEDS = [42, 30, 45, 123, 1000]
DEV_SEEDS = [42, 30, 45]

HORIZONS = {
    'japan': [3, 5, 10, 15],
    'region785': [3, 5, 10, 15],
    'state360': [3, 5, 10, 15],
    'australia-covid': [3, 7, 14],
    'nhs_timeseries': [3, 7, 14],
    'ltla_timeseries': [3, 7, 14],
}


def run_spec(dataset, horizon, seed, ablation='none', sim_mat=None,
             pprm='repeat', gate=False, target_space='level', quant=False):
    return {'dataset': dataset, 'horizon': horizon, 'seed': seed,
            'ablation': ablation, 'sim_mat': sim_mat, 'pprm': pprm,
            'gate': gate, 'target_space': target_space, 'quant': quant}


def token_for(spec):
    adj_tag = 'with_adj'
    sim_tag = f".{spec['sim_mat']}" if spec['sim_mat'] else ''
    variant_tag = ''
    if spec['pprm'] != 'repeat':
        variant_tag += f".pprm-{spec['pprm']}"
    if spec['gate']:
        variant_tag += '.sgate'
    if spec.get('target_space', 'level') != 'level':
        variant_tag += f".{spec['target_space']}"
    if spec.get('quant'):
        variant_tag += '.quant'
    return (f"MSAGAT-Net.{spec['dataset']}.w-20.h-{spec['horizon']}."
            f"{spec['ablation']}.seed-{spec['seed']}.{adj_tag}{sim_tag}{variant_tag}")


def npz_path(spec):
    return os.path.join(PRED_DIR, spec['dataset'], token_for(spec) + '.npz')


def chunk_main_uk():
    return [run_spec(d, h, s)
            for d in ('ltla_timeseries', 'nhs_timeseries')
            for h in HORIZONS[d] for s in SEEDS]


def chunk_main_influenza():
    return [run_spec(d, h, s)
            for d in ('japan', 'state360', 'region785')
            for h in HORIZONS[d] for s in SEEDS]


def chunk_main_rest():
    return [run_spec('australia-covid', h, s)
            for h in HORIZONS['australia-covid'] for s in SEEDS]


def chunk_ablation():
    specs = []
    for d in ('japan', 'ltla_timeseries', 'nhs_timeseries'):
        for h in (3, 7, 14):
            for abl in ('no_agam', 'no_mtfm', 'no_pprm'):
                for s in SEEDS:
                    specs.append(run_spec(d, h, s, ablation=abl))
    # 'none' rows for horizons the main chunks do not already cover
    for h in (7, 14):
        for s in SEEDS:
            specs.append(run_spec('japan', h, s))
    return specs


def chunk_sensitivity():
    specs = []
    for d, adj in (('ltla_timeseries', 'ltla-adj'), ('nhs_timeseries', 'nhs-adj')):
        for thr in (100, 200, 250):
            for h in HORIZONS[d]:
                for s in SEEDS:
                    specs.append(run_spec(d, h, s, sim_mat=f'{adj}-{thr}'))
    return specs


def chunk_devloop():
    """A/B the two architecture prototypes against the current model.

    Arms: control (implicitly covered by main chunks), pprm-multistep alone,
    spatial gate alone, and both combined — on the fast datasets plus an
    LTLA h=7 check, 3 seeds each.
    """
    configs = [('nhs_timeseries', 3), ('nhs_timeseries', 7),
               ('japan', 5), ('japan', 15),
               ('australia-covid', 7), ('ltla_timeseries', 7)]
    arms = [('multistep', False), ('repeat', True), ('multistep', True)]
    specs = [run_spec(d, h, s)  # control arm
             for d, h in configs for s in DEV_SEEDS]
    for pprm, gate in arms:
        for d, h in configs:
            for s in DEV_SEEDS:
                specs.append(run_spec(d, h, s, pprm=pprm, gate=gate))
    return specs


def chunk_devloop2():
    """A/B the growth-space and probabilistic variants against control.

    Control runs already exist from earlier chunks (skip-existing).
    Arms: loggrowth alone, loggrowth + quantile heads.
    """
    configs = [('nhs_timeseries', 3), ('nhs_timeseries', 7),
               ('nhs_timeseries', 14),
               ('japan', 5), ('japan', 15),
               ('australia-covid', 7), ('australia-covid', 14),
               ('region785', 15), ('ltla_timeseries', 7)]
    specs = [run_spec(d, h, s)  # control arm (mostly cached)
             for d, h in configs for s in DEV_SEEDS]
    for ts, q in (('loggrowth', False), ('loggrowth', True)):
        for d, h in configs:
            for s in DEV_SEEDS:
                specs.append(run_spec(d, h, s, target_space=ts, quant=q))
    return specs


def chunk_v2_main():
    """Full 5-seed campaign for the v2 model (loggrowth + 23-quantile heads)."""
    return [run_spec(d, h, s, target_space='loggrowth', quant=True)
            for d in HORIZONS for h in HORIZONS[d] for s in SEEDS]


CHUNKS = {
    'main_uk': chunk_main_uk,
    'main_influenza': chunk_main_influenza,
    'main_rest': chunk_main_rest,
    'ablation': chunk_ablation,
    'sensitivity': chunk_sensitivity,
    'devloop': chunk_devloop,
    'devloop2': chunk_devloop2,
    'v2_main': chunk_v2_main,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', required=True, choices=sorted(CHUNKS))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='re-run specs even when their npz already exists')
    ap.add_argument('--eval-only', action='store_true',
                    help='re-evaluate existing checkpoints without retraining')
    args = ap.parse_args()

    specs = CHUNKS[args.chunk]()
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'campaign_{args.chunk}.log')

    todo = [s for s in specs if args.force or not os.path.exists(npz_path(s))]
    print(f'chunk={args.chunk}: {len(specs)} specs, {len(specs) - len(todo)} '
          f'already done, {len(todo)} to run')
    if args.dry_run:
        for s in todo:
            print('  ' + token_for(s))
        return 0

    failed = 0
    for i, spec in enumerate(todo, 1):
        cmd = [sys.executable, '-m', 'src.train', '--single',
               '--dataset', spec['dataset'], '--horizon', str(spec['horizon']),
               '--seed', str(spec['seed']), '--ablation', spec['ablation'],
               '--pprm_supervision', spec['pprm'],
               '--target_space', spec.get('target_space', 'level')]
        if spec['sim_mat']:
            cmd += ['--sim_mat', spec['sim_mat']]
        if spec['gate']:
            cmd += ['--spatial_gate']
        if spec.get('quant'):
            cmd += ['--quantiles']
        if args.eval_only:
            cmd += ['--eval_only']

        t0 = time.time()
        print(f'[{i}/{len(todo)}] {token_for(spec)}', flush=True)
        result = subprocess.run(cmd, cwd=BASE_DIR,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)
        status = 'ok' if result.returncode == 0 else f'FAIL rc={result.returncode}'
        if result.returncode != 0:
            failed += 1
        with open(log_path, 'a', encoding='utf-8') as fh:
            fh.write(f'{time.strftime("%Y%m%d_%H%M%S")} {token_for(spec)} '
                     f'{status} {time.time() - t0:.0f}s\n')

    print(f'chunk={args.chunk} finished: {len(todo) - failed}/{len(todo)} ok, '
          f'{failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
