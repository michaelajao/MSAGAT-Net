"""The mean_agam ablation must isolate selectivity, and nothing else.

program.md requires a new experiment token be shown to change behaviour at
construction before any run is trusted. That rule exists because attention
experiment 3 once returned results bit-identical to experiment 1 --
`_init_weights` had silently undone it, caught only because the numbers were
suspiciously equal.

A first version of this ablation used a purpose-built mean-pooling module.
It was wrong: its value projection was rank-8 where EAGAM's comes from the
shared rank-24 qkv projection, so the comparison confounded capacity with
selectivity. The NHS h=3 pilot showed mean_agam 16.8% worse than the full
model while no_agam was 2.5% *better*, which is not a result about
attention at all. It was caught at 25 of 195 runs and the campaign stopped.

The current version reuses SpatialAttentionModule itself with
`uniform_attn=True`, so every parameter, projection, residual and norm is
identical and only the softmax is replaced by a fixed 1/N. These tests
assert exactly that.
"""

import os
import sys

import pytest
import torch
from argparse import Namespace

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.models import (MSAGATNet_Ablation, IdentitySpatialModule,  # noqa: E402
                        SpatialAttentionModule)

N_NODES, WINDOW, HORIZON, BATCH = 8, 20, 3, 4


class _Loader:
    """Minimal stand-in for DataBasicLoader's model-facing attributes."""

    def __init__(self, n):
        self.m = n
        self.adj = torch.eye(n) + torch.roll(torch.eye(n), 1, 0)
        self.max = torch.ones(n) * 100.0
        self.min = torch.zeros(n)


def _args(ablation):
    return Namespace(
        dataset='synthetic', horizon=HORIZON, window=WINDOW, ablation=ablation,
        hidden_dim=32, attention_heads=4, bottleneck_dim=8, num_scales=2,
        kernel_size=3, feature_channels=16, dropout=0.0,
        attention_regularization_weight=1e-5, use_adj_prior=True,
        adj_weight=0.1, use_graph_bias=True, adaptive=False, highway_window=4,
        target_space='level', quantiles=None, pprm_supervision='repeat',
        spatial_gate=False, attn_fix=False, attn_exp='', renewal=False,
        renewal_lag=0, gi_fix=None, cuda=False,
    )


def _build(ablation, seed=0):
    torch.manual_seed(seed)
    return MSAGATNet_Ablation(_args(ablation), _Loader(N_NODES))


def _forward(model, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(BATCH, WINDOW, N_NODES)
    model.eval()
    with torch.no_grad():
        out = model(x, torch.arange(BATCH))
    return out[0] if isinstance(out, tuple) else out


def test_builds_the_same_class_as_the_full_model():
    m = _build('mean_agam').graph_attention
    assert isinstance(m, SpatialAttentionModule)
    assert m.uniform_attn is True
    assert _build('none').graph_attention.uniform_attn is False
    assert isinstance(_build('no_agam').graph_attention, IdentitySpatialModule)


def test_parameters_are_identical_to_the_full_model():
    """This is what the first attempt got wrong: same capacity, same init."""
    full = dict(_build('none', seed=3).graph_attention.named_parameters())
    mean = dict(_build('mean_agam', seed=3).graph_attention.named_parameters())
    assert set(full) == set(mean), 'parameter names differ'
    for k in full:
        assert full[k].shape == mean[k].shape, f'{k} shape differs'
        assert torch.equal(full[k], mean[k]), f'{k} initialised differently'


def test_attention_is_exactly_uniform_and_normalised():
    m = _build('mean_agam')
    _forward(m)
    a = m.graph_attention.attn
    assert torch.allclose(a, torch.full_like(a, 1.0 / N_NODES))
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)))


def test_predictions_differ_from_both_neighbours():
    """It must be neither a no-op nor a rename of no_agam."""
    full = _forward(_build('none'))
    mean = _forward(_build('mean_agam'))
    ident = _forward(_build('no_agam'))
    assert not torch.allclose(mean, full), 'mean_agam is identical to full attention'
    assert not torch.allclose(mean, ident), 'mean_agam is identical to no_agam'


def test_the_softmax_is_the_only_difference():
    """Force the full model's attention uniform: it must then match exactly.

    This is the property that makes the ablation interpretable. If forcing
    the flag reproduces the mean_agam output bit-for-bit, then any measured
    difference between the two arms is attributable to attention
    selectivity and to nothing else.
    """
    full = _build('none', seed=7)
    mean = _build('mean_agam', seed=7)
    before = _forward(full)
    full.graph_attention.uniform_attn = True          # flip only the softmax
    after = _forward(full)
    assert not torch.allclose(before, after), 'flipping the flag changed nothing'
    assert torch.allclose(after, _forward(mean), atol=1e-6), (
        'uniform-forced full model does not match mean_agam, so the two arms '
        'differ by more than the softmax')


@pytest.mark.parametrize('ablation', ['none', 'no_agam', 'mean_agam'])
def test_forward_shape(ablation):
    assert _forward(_build(ablation)).shape == (BATCH, HORIZON, N_NODES)
