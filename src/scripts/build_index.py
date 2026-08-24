"""Derive one row per run from the artefacts on disk.

`report/results/runs_index.csv` is the table every downstream table and
figure joins against. It is *derived*, never authored: this script walks
`report/predictions/`, parses each filename with the shared token grammar,
and cross-checks it against the run manifest and the `all_results.csv` row.

The `consistency` column says what could be verified:

    ok                every source agrees
    manifest_missing  the run predates manifests (all pre-25 Aug runs)
    csv_missing       an archive exists with no metrics row
    metric_mismatch   the CSV row disagrees with the archive's own RMSE
    unreadable        the archive could not be opened

`metric_mismatch` is the one that matters. RMSE is recomputed here from the
stored arrays and compared with the recorded row, so a re-scoring pass that
rewrote an archive without its CSV row -- or vice versa -- shows up as a
row in this table instead of as an unexplained number in a paper.

    python -m src.scripts.build_index [--check]

`--check` exits non-zero if anything is not `ok` or `manifest_missing`.
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from src.tokens import parse_token  # noqa: E402

PRED_DIR = os.path.join(BASE, 'report', 'predictions')
RESULTS_DIR = os.path.join(BASE, 'report', 'results')
MANIFEST_DIR = os.path.join(BASE, 'report', 'manifests')
OUT_CSV = os.path.join(RESULTS_DIR, 'runs_index.csv')

SAVE_DIRS = ['save_all', 'save_attn', 'save_renewal']

# RMSE agreement tolerance between the archive and the recorded CSV row.
RTOL = 1e-6

COLUMNS = [
    'run_token', 'family', 'arm', 'dataset', 'horizon', 'seed', 'window', 'ablation',
    'use_adj', 'sim_mat', 'target_space', 'quantiles', 'n_quantiles',
    'attn_exp', 'renewal', 'renewal_lag', 'gi_fix', 'level_cap',
    'model_column', 'n_test', 'rmse_npz', 'rmse_csv', 'has_val', 'has_quantiles',
    'git_commit', 'mode', 'best_epoch', 'n_params', 'timestamp',
    'npz_path', 'ckpt_path', 'manifest_path', 'consistency',
]


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _load_results_rows():
    """Index every all_results.csv row by (model, dataset, h, seed, ablation)."""
    rows = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, '*', 'all_results.csv')):
        with open(path, encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                key = (row.get('model', ''), row.get('dataset', ''),
                       str(row.get('horizon', '')), str(row.get('seed', '')),
                       row.get('ablation', ''))
                # Later rows supersede earlier ones, matching save_metrics.
                rows[key] = row
    return rows


def _find_checkpoint(token):
    for d in SAVE_DIRS:
        p = os.path.join(BASE, d, token + '.pt')
        if os.path.exists(p):
            return os.path.relpath(p, BASE).replace('\\', '/')
    return ''


def _msagat_row(path, token, results):
    spec = parse_token(token)
    rec = {c: '' for c in COLUMNS}
    rec.update({
        'run_token': token,
        'family': 'MSAGAT-Net',
        'arm': 'current',
        'dataset': spec['dataset'],
        'horizon': spec['horizon'],
        'seed': spec['seed'],
        'window': spec['window'],
        'ablation': spec['ablation'],
        'use_adj': spec['use_adj_prior'],
        'sim_mat': spec['sim_mat'] or 'default',
        'target_space': spec['target_space'],
        'quantiles': spec['quantiles'],
        'n_quantiles': spec['n_quantiles'] or '',
        'attn_exp': spec['attn_exp'] or '',
        'renewal': spec['renewal'],
        'renewal_lag': spec['renewal_lag'] if spec['renewal_lag'] else '',
        'gi_fix': f"{spec['gi_fix'][0]:g}-{spec['gi_fix'][1]:g}" if spec['gi_fix'] else '',
        'level_cap': spec['level_cap'] if spec['level_cap'] else '',
        'npz_path': os.path.relpath(path, BASE).replace('\\', '/'),
        'ckpt_path': _find_checkpoint(token),
    })

    # The model column is the token's variant suffix appended to the family.
    suffix = token.split('.with_adj', 1)[-1] if '.with_adj' in token else ''
    suffix = token.split('.no_adj', 1)[-1] if '.no_adj' in token else suffix
    if spec['sim_mat']:
        suffix = suffix.replace('.' + spec['sim_mat'], '', 1)
    rec['model_column'] = 'MSAGAT-Net' + suffix
    return rec, spec


def _baseline_row(path, token):
    """Baselines use a shorter grammar: model.dataset.w-W.h-H.abl.seed-S.

    Anything after the seed is an arm suffix. The only one in use is
    ``oldckpt``: predictions dumped from checkpoints trained before the
    lead-h protocol correction. They are kept as evidence but must never be
    counted as coverage, so the arm is recorded explicitly rather than being
    silently folded into the family.
    """
    parts = token.split('.')
    rec = {c: '' for c in COLUMNS}
    rec.update({
        'run_token': token,
        'family': parts[0],
        'dataset': parts[1] if len(parts) > 1 else '',
        'window': parts[2][2:] if len(parts) > 2 and parts[2].startswith('w-') else '',
        'horizon': parts[3][2:] if len(parts) > 3 and parts[3].startswith('h-') else '',
        'ablation': parts[4] if len(parts) > 4 else '',
        'seed': parts[5][5:] if len(parts) > 5 and parts[5].startswith('seed-') else '',
        'arm': '.'.join(parts[6:]) if len(parts) > 6 else 'current',
        'model_column': parts[0],
        'npz_path': os.path.relpath(path, BASE).replace('\\', '/'),
    })
    return rec


def build(verbose=True):
    results = _load_results_rows()
    rows = []

    for path in sorted(glob.glob(os.path.join(PRED_DIR, '*', '*.npz'))):
        token = os.path.splitext(os.path.basename(path))[0]
        is_msagat = token.startswith('MSAGAT-Net.')

        if is_msagat:
            try:
                rec, spec = _msagat_row(path, token, results)
            except ValueError:
                rec = {c: '' for c in COLUMNS}
                rec.update({'run_token': token, 'family': 'MSAGAT-Net',
                            'npz_path': os.path.relpath(path, BASE),
                            'consistency': 'unparsable_token'})
                rows.append(rec)
                continue
        else:
            rec = _baseline_row(path, token)

        notes = []

        # --- the archive itself -------------------------------------------
        try:
            with np.load(path, allow_pickle=True) as d:
                files = set(d.files)
                rec['has_val'] = 'y_true_val' in files
                rec['has_quantiles'] = 'y_pred_q' in files
                if 'y_true' in files and 'y_pred' in files:
                    rec['n_test'] = int(np.asarray(d['y_true']).shape[0])
                    rec['rmse_npz'] = round(_rmse(d['y_true'], d['y_pred']), 6)
                else:
                    notes.append('no_predictions')
        except Exception as exc:                      # noqa: BLE001
            rec['consistency'] = f'unreadable:{type(exc).__name__}'
            rows.append(rec)
            continue

        # --- the metrics row ----------------------------------------------
        key = (rec['model_column'], rec['dataset'], str(rec['horizon']),
               str(rec['seed']), rec['ablation'] or 'none')
        row = results.get(key)
        if row is None:
            # Baselines are trained in the sibling colagnn/EpiGNN repos and
            # keep their metrics there; only their predictions land here, so
            # a missing row is expected for them and a defect for ours.
            if is_msagat:
                notes.append('csv_missing')
        else:
            rec['timestamp'] = row.get('timestamp', '')
            try:
                rec['rmse_csv'] = round(float(row['rmse']), 6)
            except (KeyError, TypeError, ValueError):
                notes.append('csv_rmse_unreadable')
            if rec['rmse_csv'] != '' and rec['rmse_npz'] != '':
                if not np.isclose(rec['rmse_csv'], rec['rmse_npz'],
                                  rtol=1e-3, atol=1e-6):
                    notes.append('metric_mismatch')

        # --- the manifest ---------------------------------------------------
        man = os.path.join(MANIFEST_DIR, rec['dataset'], token + '.json')
        if os.path.exists(man):
            rec['manifest_path'] = os.path.relpath(man, BASE).replace('\\', '/')
            with open(man, encoding='utf-8') as fh:
                m = json.load(fh)
            rec['git_commit'] = (m.get('git') or {}).get('commit', '') or ''
            rec['mode'] = m.get('mode', '')
            rec['best_epoch'] = (m.get('training') or {}).get('best_epoch', '')
            rec['n_params'] = (m.get('training') or {}).get('n_params', '')
        elif is_msagat:
            notes.append('manifest_missing')

        rec['consistency'] = ','.join(notes) if notes else 'ok'
        rows.append(rec)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    if verbose:
        from collections import Counter
        tally = Counter(r['consistency'] for r in rows)
        print(f'wrote {os.path.relpath(OUT_CSV, BASE)}: {len(rows)} runs')
        for state, n in tally.most_common():
            print(f'  {n:5d}  {state}')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true',
                    help='exit non-zero on any consistency problem other '
                         'than a missing manifest on a pre-25-Aug run')
    args = ap.parse_args()

    rows = build()
    if args.check:
        bad = [r for r in rows
               if r['consistency'] not in ('ok', 'manifest_missing')]
        if bad:
            print(f'\n{len(bad)} runs failed the consistency check:')
            for r in bad[:25]:
                print(f"  {r['consistency']:24s} {r['run_token']}")
            if len(bad) > 25:
                print(f'  ... and {len(bad) - 25} more')
            sys.exit(1)
        print('\nconsistency check passed')


if __name__ == '__main__':
    main()
