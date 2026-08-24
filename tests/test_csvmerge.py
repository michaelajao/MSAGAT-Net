"""Merge-on-write must never lose rows it did not recompute.

The failure this guards against is concrete: `conformal.py` and
`prob_eval.py` both ended with `df.to_csv(OUT_CSV)`, so
`--dataset nhs_timeseries` replaced all 450 conformal rows with about four,
silently, leaving a file that looked complete.
"""

import os
import sys

import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.csvmerge import merge_rows  # noqa: E402

KEYS = ['dataset', 'horizon', 'seed', 'method']


def _seed_file(path):
    rows = [{'dataset': d, 'horizon': h, 'seed': s, 'method': m, 'wis': 1.0}
            for d in ('nhs', 'ltla') for h in (3, 7)
            for s in (42, 30) for m in ('raw', 'identity')]
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)


def test_creates_when_absent(tmp_path):
    out = tmp_path / 'm.csv'
    merge_rows(str(out), [{'dataset': 'nhs', 'horizon': 3, 'seed': 42,
                           'method': 'raw', 'wis': 2.0}], keys=KEYS,
               verbose=False)
    assert len(pd.read_csv(out)) == 1


def test_partial_run_keeps_every_other_row(tmp_path):
    out = tmp_path / 'm.csv'
    n = _seed_file(out)
    merge_rows(str(out), [{'dataset': 'nhs', 'horizon': 3, 'seed': 42,
                           'method': 'raw', 'wis': 99.0}], keys=KEYS,
               verbose=False)
    after = pd.read_csv(out)
    assert len(after) == n, 'a filtered run must not shrink the file'
    hit = after[(after.dataset == 'nhs') & (after.horizon == 3)
                & (after.seed == 42) & (after.method == 'raw')]
    assert len(hit) == 1 and hit.wis.iloc[0] == 99.0


def test_replace_all_is_explicit(tmp_path):
    out = tmp_path / 'm.csv'
    _seed_file(out)
    merge_rows(str(out), [{'dataset': 'nhs', 'horizon': 3, 'seed': 42,
                           'method': 'raw', 'wis': 5.0}], keys=KEYS,
               replace_all=True, verbose=False)
    assert len(pd.read_csv(out)) == 1


def test_schema_change_preserves_the_old_file(tmp_path):
    out = tmp_path / 'm.csv'
    _seed_file(out)
    merge_rows(str(out), [{'dataset': 'nhs', 'horizon': 3, 'seed': 42,
                           'variant': 'v2', 'wis': 5.0}],
               keys=['dataset', 'horizon', 'seed', 'variant'], verbose=False)
    assert os.path.exists(str(out) + '.superseded')


def test_missing_key_column_raises(tmp_path):
    out = tmp_path / 'm.csv'
    with pytest.raises(KeyError):
        merge_rows(str(out), [{'dataset': 'nhs'}], keys=KEYS, verbose=False)


def test_empty_rows_is_a_noop(tmp_path):
    out = tmp_path / 'm.csv'
    n = _seed_file(out)
    merge_rows(str(out), [], keys=KEYS, verbose=False)
    assert len(pd.read_csv(out)) == n
