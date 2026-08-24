"""Round-trip the token grammar against every artefact already on disk.

This is the safety proof for consolidating eight token builders into one:
if `parse_token` then `build_token` reproduces every existing filename
byte-for-byte, no npz, checkpoint or attention dump can be orphaned by the
change. Baseline predictions (cola_gnn, lstnet, dcrnn, epignn, CNNRNN_Res)
use a different, simpler grammar and are skipped by name.
"""

import glob
import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.tokens import build_token, parse_token  # noqa: E402

PRED_DIR = os.path.join(BASE, 'report', 'predictions')
SAVE_DIRS = ['save_all', 'save_attn', 'save_renewal']


def _msagat_files(pattern, root):
    """Every MSAGAT-Net artefact under `root`, baselines excluded."""
    hits = glob.glob(os.path.join(root, pattern), recursive=True)
    return [h for h in hits if os.path.basename(h).startswith('MSAGAT-Net.')]


def _all_tokens():
    names = []
    names += _msagat_files(os.path.join('**', '*.npz'), PRED_DIR)
    for d in SAVE_DIRS:
        names += _msagat_files('*.pt', os.path.join(BASE, d))
    names += _msagat_files('*.npy', os.path.join(BASE, 'report', 'attention'))
    stems = {os.path.splitext(os.path.basename(n))[0] for n in names}
    return sorted(stems)


TOKENS = _all_tokens()


def test_corpus_is_not_empty():
    """Guard against the test silently passing on zero files."""
    assert len(TOKENS) > 100, f'only {len(TOKENS)} tokens found under {BASE}'


@pytest.mark.parametrize('token', TOKENS)
def test_roundtrip(token):
    """parse -> build reproduces the filename exactly."""
    assert build_token(**parse_token(token)) == token


def test_parse_rejects_rubbish():
    with pytest.raises(ValueError):
        parse_token('not-a-token')
    with pytest.raises(ValueError):
        parse_token('MSAGAT-Net.japan.w-20.h-3.none.seed-notanumber.with_adj')
    # A trailing segment that is neither a known variant nor a leading
    # sim_mat must fail loudly rather than be dropped.
    with pytest.raises(ValueError):
        parse_token('MSAGAT-Net.japan.w-20.h-3.none.seed-42.with_adj.quant.bogus')


def test_sim_mat_roundtrips():
    t = build_token(dataset='ltla_timeseries', horizon=7, seed=42,
                    sim_mat='ltla-adj-200')
    assert '.ltla-adj-200' in t
    assert parse_token(t)['sim_mat'] == 'ltla-adj-200'
    assert build_token(**parse_token(t)) == t


def test_legacy_tokens_have_no_adjacency_segment():
    """Pre-train.py runs end at seed-N; they must round-trip unchanged."""
    legacy = 'MSAGAT-Net.ltla_timeseries.w-20.h-3.none.seed-5'
    spec = parse_token(legacy)
    assert spec['use_adj_prior'] is None
    assert build_token(**spec) == legacy


def test_new_segments_are_noops_at_defaults():
    """The .q and .cap segments must not appear at historical defaults."""
    base = dict(dataset='nhs_timeseries', horizon=7, seed=42,
                target_space='loggrowth', quantiles=True)
    assert build_token(**base) == build_token(n_quantiles=23, level_cap=3.0, **base)
    assert '.q23' not in build_token(n_quantiles=23, **base)
    assert '.cap3' not in build_token(level_cap=3.0, **base)
    # ...and that they do appear when the value actually differs.
    assert '.q7' in build_token(n_quantiles=7, **base)
    assert '.cap2.5' in build_token(level_cap=2.5, **base)


def test_new_segments_roundtrip():
    t = build_token(dataset='nhs_timeseries', horizon=7, seed=42,
                    target_space='loggrowth', quantiles=True,
                    n_quantiles=7, level_cap=2.5)
    assert build_token(**parse_token(t)) == t
