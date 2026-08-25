"""The mean_agam ablation must actually change the model.

program.md requires that a new experiment token be shown to change
behaviour at construction before any run is trusted. That rule exists
because attention experiment 3 once returned results bit-identical to
experiment 1: `_init_weights` had silently undone the change, and it was
only caught because the numbers were suspiciously equal.

These tests assert that `mean_agam` builds a distinct module, that its
attention is exactly uniform, and that it produces different predictions
from both the full attention and the `no_agam` identity.
"""

import os
import sys

import pytest
import torch
from argparse import Namespace

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.models import (MSAGATNet_Ablation, MeanPoolSpatialModule,  # noqa: E402
                        IdentitySpatialModule, SpatialAttentionModule)

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


def test_builds_the_right_module():
    assert isinstance(_build('mean_agam').graph_attention, MeanPoolSpatialModule)
    assert isinstance(_build('none').graph_attention, SpatialAttentionModule)
    assert isinstance(_build('no_agam').graph_attention, IdentitySpatialModule)


def test_attention_is_exactly_uniform():
    m = _build('mean_agam')
    _forward(m)
    a = m.graph_attention.attn
    assert torch.allclose(a, torch.full_like(a, 1.0 / N_NODES))
    # Rows must still be a distribution.
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)))


def test_predictions_differ_from_both_neighbours():
    """The whole point: it must not be a no-op or a rename of no_agam."""
    full = _forward(_build('none'))
    mean = _forward(_build('mean_agam'))
    ident = _forward(_build('no_agam'))
    assert not torch.allclose(mean, full), 'mean_agam is identical to full attention'
    assert not torch.allclose(mean, ident), 'mean_agam is identical to no_agam'


def test_pooling_is_permutation_invariant():
    """A uniform mean cannot depend on node order; a selective one can."""
    m = _build('mean_agam')
    torch.manual_seed(1)
    x = torch.rand(BATCH, N_NODES, 32)
    perm = torch.randperm(N_NODES)
    m.eval()
    with torch.no_grad():
        a, _ = m.graph_attention(x)
        b, _ = m.graph_attention(x[:, perm])
    # The pooled contribution is order-free, so permuting the input permutes
    # the output exactly.
    assert torch.allclose(a[:, perm], b, atol=1e-6)


def test_parameter_count_is_smaller_than_full_attention():
    """It drops the query/key projections and the learnable graph bias."""
    full = sum(p.numel() for p in _build('none').graph_attention.parameters())
    mean = sum(p.numel() for p in _build('mean_agam').graph_attention.parameters())
    assert mean < full, (mean, full)


@pytest.mark.parametrize('ablation', ['none', 'no_agam', 'mean_agam'])
def test_forward_shape(ablation):
    out = _forward(_build(ablation))
    assert out.shape == (BATCH, HORIZON, N_NODES)
