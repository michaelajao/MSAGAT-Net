"""Generate every LaTeX table for the evaluation paper from artefacts.

All seven tables in the submitted manuscript were hand-typed; there is no
script anywhere that emits an Elsevier table, and the numbers in them are the
ones finding E1 invalidated. Nothing here is typed: every value is read from
`report/results/` or from the prediction archives, so regenerating the paper
regenerates its numbers.

    python -m src.scripts.paperB_tables

Writes bare `tabular` fragments to `doc/elservier/tables/*.tex` for
`\\input{}`, plus `_values.json` recording every number the prose quotes so
the text can be checked against the tables mechanically.

Backslashes are built through the BS constant rather than written literally:
the shell heredocs used to drive this repository collapse them, and that has
silently corrupted generated LaTeX here before.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from src.tokens import parse_token  # noqa: E402

RESULTS = os.path.join(BASE, 'report', 'results')
OUT = os.path.join(BASE, 'doc', 'elservier', 'tables')

BS = chr(92)
EOL = BS + BS
HLINE = BS + 'hline'

DATASETS = ['japan', 'region785', 'state360',
            'australia-covid', 'ltla_timeseries', 'nhs_timeseries']
PRETTY = {'japan': 'Japan-Prefectures', 'region785': 'US-Regions',
          'state360': 'US-States', 'australia-covid': 'Australia-COVID',
          'ltla_timeseries': 'UK-LTLA', 'nhs_timeseries': 'NHS-Regions'}
BASELINES = ['cola_gnn', 'epignn', 'dcrnn', 'lstnet', 'CNNRNN_Res']
BASE_PRETTY = {'cola_gnn': 'Cola-GNN', 'epignn': 'EpiGNN', 'dcrnn': 'DCRNN',
               'lstnet': 'LSTNet', 'CNNRNN_Res': 'CNNRNN-Res'}
FLOORS = ['persistence', 'seasonal_naive', 'ar4']
FLOOR_PRETTY = {'persistence': 'Persistence', 'seasonal_naive': 'Seasonal-naive',
                'ar4': 'AR(4)'}


def write(name, lines):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'  wrote {os.path.relpath(path, BASE)}')


def index():
    """Current-protocol runs, excluding the attnfix attention experiment.

    attnfix carries no attn_exp token, so an attn_exp filter alone admits it
    into the frozen v2 arm.
    """
    df = pd.read_csv(os.path.join(RESULTS, 'runs_index.csv'))
    df = df[(df.arm == 'current') & df.rmse_npz.notna()]
    if 'attn_fix' in df.columns:
        df = df[df.attn_fix != True]                              # noqa: E712
    return df


def msagat_v2(df):
    """The paper's model: log-growth targets, quantile heads, direct decoder."""
    return df[(df.family == 'MSAGAT-Net') & (df.target_space == 'loggrowth')
              & (df.quantiles == True) & df.attn_exp.isna()        # noqa: E712
              & (df.renewal == False) & (df.ablation == 'none')    # noqa: E712
              & (df.sim_mat == 'default')]


def fmt(mean, sd, bold=False, prec=2):
    body = f'{mean:.{prec}f} {BS}pm {sd:.{prec}f}'
    return ('$' + BS + 'mathbf{' + body + '}$') if bold else f'${body}$'


# --------------------------------------------------------------------------
# T1  datasets
# --------------------------------------------------------------------------
def table_datasets(values):
    rows = []
    for ds in DATASETS:
        raw = np.loadtxt(os.path.join(BASE, 'data', f'{ds}.txt'), delimiter=',')
        n, m = raw.shape
        adj_name = {'japan': 'japan-adj', 'region785': 'region-adj',
                    'state360': 'state-adj-49', 'australia-covid': 'australia-adj',
                    'ltla_timeseries': 'ltla-adj', 'nhs_timeseries': 'nhs-adj'}[ds]
        A = np.loadtxt(os.path.join(BASE, 'data', f'{adj_name}.txt'), delimiter=',')
        density = A.mean()
        integer = np.allclose(raw, np.round(raw))
        rows.append((PRETTY[ds], m, n, 'weekly' if ds in
                     ('japan', 'region785', 'state360') else 'daily',
                     density, 'raw' if integer else '7-day mean'))
        values[f'n_nodes_{ds}'] = int(m)
        values[f'n_steps_{ds}'] = int(n)

    lines = [BS + 'begin{tabular}{lrrlrl}', HLINE,
             ' & '.join(['Dataset', 'Regions', 'Steps', 'Resolution',
                         'Graph density', 'Smoothing']) + ' ' + EOL, HLINE]
    for name, m, n, res, dens, sm in rows:
        lines.append(f'{name} & {m} & {n} & {res} & {dens:.3f} & {sm} ' + EOL)
    lines += [HLINE, BS + 'end{tabular}']
    write('datasets.tex', lines)


# --------------------------------------------------------------------------
# T2/T3  main comparison with DM markers
# --------------------------------------------------------------------------
def table_main(df, values, group, name):
    dm = pd.read_csv(os.path.join(RESULTS, 'dm_tests_v2_best.csv'))
    dm = dm[dm.family == 'baselines']
    ours = msagat_v2(df)

    horizons = sorted(df[df.dataset == group[0]].horizon.unique())
    header = ['Method'] + [f'$h={h}$' for h in horizons]

    lines = [BS + 'begin{tabular}{l' + 'c' * len(horizons) + '}', HLINE]
    for ds in group:
        hs = sorted(df[df.dataset == ds].horizon.unique())
        lines.append(BS + 'multicolumn{' + str(len(hs) + 1) + '}{l}{'
                     + BS + 'textbf{' + PRETTY[ds] + '}} ' + EOL)
        lines.append(' & '.join(['Method'] + [f'$h={h}$' for h in hs])
                     + ' ' + EOL)
        # cell -> best mean for bolding
        best = {}
        for h in hs:
            cands = []
            g = ours[(ours.dataset == ds) & (ours.horizon == h)]
            if not g.empty:
                cands.append(g.rmse_npz.mean())
            for b in BASELINES:
                gb = df[(df.family == b) & (df.dataset == ds) & (df.horizon == h)]
                if not gb.empty:
                    cands.append(gb.rmse_npz.mean())
            best[h] = min(cands) if cands else np.nan

        for b in BASELINES:
            cells = []
            for h in hs:
                gb = df[(df.family == b) & (df.dataset == ds) & (df.horizon == h)]
                if gb.empty:
                    cells.append('--')
                    continue
                mu, sd = gb.rmse_npz.mean(), gb.rmse_npz.std(ddof=1)
                cells.append(fmt(mu, sd, bold=np.isclose(mu, best[h])))
            lines.append(f'{BASE_PRETTY[b]} & ' + ' & '.join(cells) + ' ' + EOL)

        cells = []
        for h in hs:
            g = ours[(ours.dataset == ds) & (ours.horizon == h)]
            if g.empty:
                cells.append('--')
                continue
            mu, sd = g.rmse_npz.mean(), g.rmse_npz.std(ddof=1)
            cell = fmt(mu, sd, bold=np.isclose(mu, best[h]))
            d = dm[(dm.dataset == ds) & (dm.horizon == h) & dm.significant]
            wins = (d.direction == 'MSAGAT better').sum()
            losses = (d.direction == 'baseline better').sum()
            mark = ''
            if wins:
                mark += BS + 'ensuremath{^{' + BS + 'triangle ' + str(wins) + '}}'
            if losses:
                mark += BS + 'ensuremath{_{' + BS + 'triangledown ' + str(losses) + '}}'
            cells.append(cell + mark)
            values[f'rmse_{ds}_h{h}'] = round(float(mu), 4)
        lines.append('MSAGAT-Net & ' + ' & '.join(cells) + ' ' + EOL)
        lines.append(HLINE)

    lines.append(BS + 'end{tabular}')
    write(name, lines)


# --------------------------------------------------------------------------
# T4  naive floors
# --------------------------------------------------------------------------
def table_floors(values):
    fc = pd.read_csv(os.path.join(RESULTS, 'floor_comparison.csv'))
    lines = [BS + 'begin{tabular}{llrrrrl}', HLINE,
             ' & '.join(['Dataset', '$h$', 'MSAGAT-Net', 'Best baseline',
                         'Best floor', 'Floor', 'Beaten?']) + ' ' + EOL, HLINE]
    for _, r in fc.iterrows():
        beaten = 'yes' if r.any_trained_beats_floor else BS + 'textbf{no}'
        lines.append(
            f'{PRETTY.get(r.dataset, r.dataset)} & {int(r.horizon)} & '
            f'{r.msagat_v2:.2f} & {r.best_baseline:.2f} & {r.best_floor:.2f} & '
            f'{FLOOR_PRETTY.get(r.best_floor_name, r.best_floor_name)} & '
            f'{beaten} ' + EOL)
    lines += [HLINE, BS + 'end{tabular}']
    write('floors.tex', lines)
    values['floors_v2_beats'] = int(fc.v2_beats_floor.sum())
    values['floors_any_beats'] = int(fc.any_trained_beats_floor.sum())
    values['floors_n_cells'] = int(len(fc))


# --------------------------------------------------------------------------
# T5  ablations
# --------------------------------------------------------------------------
def table_ablation(values):
    path = os.path.join(RESULTS, 'ablation_v2.csv')
    if not os.path.exists(path):
        print('  (ablation_v2.csv missing, skipping)')
        return
    ab = pd.read_csv(path)
    arms = ['mean_agam', 'no_agam', 'no_mtfm', 'no_pprm']
    label = {'mean_agam': 'Uniform attention (softmax removed)',
             'no_agam': 'No spatial attention', 'no_mtfm': 'No multi-hop refinement',
             'no_pprm': 'No progressive refinement'}
    lines = [BS + 'begin{tabular}{lrrrr}', HLINE,
             ' & '.join(['Component removed', 'Median $' + BS + 'Delta$RMSE',
                         'Worst cell', 'Cells worse', 'Cells']) + ' ' + EOL, HLINE]
    for a in arms:
        g = ab[ab.ablation == a]
        if g.empty:
            continue
        med = g.delta_pct_median.median()
        lines.append(f'{label[a]} & {med:+.2f}{BS}% & '
                     f'{g.delta_pct_median.max():+.2f}{BS}% & '
                     f'{int((g.delta_pct_median > 0).sum())} & {len(g)} ' + EOL)
        values[f'ablation_{a}_median'] = round(float(med), 3)
    lines += [HLINE, BS + 'end{tabular}']
    write('ablation.tex', lines)

    ltla = ab[(ab.ablation == 'mean_agam') & (ab.dataset == 'ltla_timeseries')]
    if not ltla.empty:
        values['mean_agam_ltla_mean_abs'] = round(
            float(ltla.delta_pct_mean.abs().mean()), 3)
        values['mean_agam_ltla_min_p'] = round(float(ltla.wilcoxon_p.min()), 3)


# --------------------------------------------------------------------------
# T6  calibration
# --------------------------------------------------------------------------
def table_calibration(values):
    cs = pd.read_csv(os.path.join(RESULTS, 'calibration_summary.csv'))
    methods = ['raw', 'identity', 'adjacency', 'uniform', 'attention']
    label = {'raw': 'Uncalibrated', 'identity': 'Per-region conformal',
             'adjacency': 'Adjacency-weighted', 'uniform': 'Uniformly pooled',
             'attention': 'Attention-weighted'}
    lines = [BS + 'begin{tabular}{lrrrr}', HLINE,
             ' & '.join(['Calibration', '50{' + BS + '%} cov.',
                         '90{' + BS + '%} cov.', '95{' + BS + '%} cov.',
                         'WIS']) + ' ' + EOL, HLINE]
    for m in methods:
        g = cs[cs.method == m]
        if g.empty:
            continue
        lines.append(f'{label[m]} & {g.cov50.mean():.3f} & {g.cov90.mean():.3f} & '
                     f'{g.cov95.mean():.3f} & {g.wis.mean():.1f} ' + EOL)
        values[f'cov90_{m}'] = round(float(g.cov90.mean()), 4)
    lines += [HLINE, BS + 'end{tabular}']
    write('calibration.tex', lines)


# --------------------------------------------------------------------------
# T7  minimum detectable effect
# --------------------------------------------------------------------------
def table_power(values):
    pw = pd.read_csv(os.path.join(RESULTS, 'dm_power_v2_best.csv'))
    pw = pw[pw.family == 'baselines']
    g = (pw.groupby(['dataset', 'horizon'])
           .agg(n_test=('n_test', 'first'), mde=('mde_rmse_pct', 'median'))
           .reset_index().sort_values('mde', ascending=False))
    lines = [BS + 'begin{tabular}{lrrr}', HLINE,
             ' & '.join(['Dataset', '$h$', 'Test points',
                         'Min. detectable ' + BS + 'Delta RMSE']) + ' ' + EOL,
             HLINE]
    for _, r in g.iterrows():
        mde = '--' if not np.isfinite(r.mde) else f'{r.mde:.1f}{BS}%'
        lines.append(f'{PRETTY.get(r.dataset, r.dataset)} & {int(r.horizon)} & '
                     f'{int(r.n_test)} & {mde} ' + EOL)
    lines += [HLINE, BS + 'end{tabular}']
    write('power.tex', lines)
    q = pw.mde_rmse_pct.dropna()
    values['mde_median'] = round(float(q.median()), 2)
    values['mde_min'] = round(float(q.min()), 2)
    values['mde_max'] = round(float(q.max()), 2)


# --------------------------------------------------------------------------
# T8  pooled-symmetric appendix
# --------------------------------------------------------------------------
def table_pooled(values):
    path = os.path.join(RESULTS, 'pooled_symmetric.csv')
    if not os.path.exists(path):
        return
    ps = pd.read_csv(path)
    agg = (ps.groupby(['dataset', 'horizon', 'arm'])
             .agg(lead_h=('rmse_lead_h', 'mean'), pooled=('rmse_pooled', 'mean'))
             .reset_index())
    agg['penalty'] = 100 * (agg.pooled - agg.lead_h) / agg.lead_h
    lines = [BS + 'begin{tabular}{llrrr}', HLINE,
             ' & '.join(['Dataset', '$h$ / arm', 'Lead-$h$', 'Pooled',
                         'Penalty']) + ' ' + EOL, HLINE]
    for _, r in agg.iterrows():
        lines.append(f'{PRETTY.get(r.dataset, r.dataset)} & '
                     f'{int(r.horizon)} / {r.arm} & {r.lead_h:.2f} & '
                     f'{r.pooled:.2f} & {r.penalty:+.1f}{BS}% ' + EOL)
    lines += [HLINE, BS + 'end{tabular}']
    write('pooled_symmetric.tex', lines)


def main():
    df = index()
    values = {}
    print('generating Paper B tables from artefacts:')
    table_datasets(values)
    table_main(df, values, ['japan', 'region785', 'state360'], 'main_influenza.tex')
    table_main(df, values, ['australia-covid', 'ltla_timeseries',
                            'nhs_timeseries'], 'main_covid.tex')
    table_floors(values)
    table_ablation(values)
    table_calibration(values)
    table_power(values)
    table_pooled(values)

    dm = pd.read_csv(os.path.join(RESULTS, 'dm_tests_v2_best.csv'))
    b = dm[dm.family == 'baselines']
    values['dm_n'] = int(len(b))
    values['dm_wins'] = int((b.significant & (b.direction == 'MSAGAT better')).sum())
    values['dm_losses'] = int((b.significant & (b.direction == 'baseline better')).sum())
    values['dm_ties'] = int((~b.significant).sum())

    with open(os.path.join(OUT, '_values.json'), 'w', encoding='utf-8') as fh:
        json.dump(values, fh, indent=2, sort_keys=True)
    print(f'  wrote {os.path.relpath(os.path.join(OUT, "_values.json"), BASE)}'
          f' ({len(values)} values the prose may quote)')


if __name__ == '__main__':
    main()
