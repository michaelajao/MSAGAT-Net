"""The one definition of the run-token grammar.

A run token is the only identity an artefact carries: it names the prediction
npz, the checkpoint, the attention dump and the TensorBoard directory, and it
is reconstructed as the ``model`` column of ``all_results.csv``. Until now the
grammar was re-implemented in eight places and parsed by five regexes, and at
least one copy (``campaign.py.token_for``) had already diverged by omitting
the renewal and generation-interval suffixes.

    MSAGAT-Net.{dataset}.w-{window}.h-{horizon}.{ablation}.seed-{seed}.{adj}{sim}{variant}

``build_token`` reproduces ``src/train.py:551-571`` exactly, so every one of
the artefacts already on disk round-trips through this module unchanged;
``tests/test_tokens.py`` asserts that against every filename in
``report/predictions/``. Do not change the grammar without running it.

Two optional segments are new and are no-ops for every historical run, so
nothing on disk is renamed:

``.q{n}``    when the quantile count differs from the 23-level FluSight set
``.cap{c}``  when the growth level cap differs from 3.0

Both exist because a run that changes either produces different numbers under
an identical filename and an identical CSV dedup key -- the collision that
made the 14 August re-scoring undetectable after the fact.
"""

import os
import re

__all__ = [
    'DEFAULT_N_QUANTILES', 'DEFAULT_LEVEL_CAP',
    'build_token', 'build_variant_tag', 'parse_token',
    'npz_path', 'ckpt_path', 'attention_path', 'manifest_path',
]

DEFAULT_N_QUANTILES = 23
DEFAULT_LEVEL_CAP = 3.0

MODEL_NAME = 'MSAGAT-Net'


def build_token(dataset, horizon, seed, window=20, ablation='none',
                use_adj_prior=True, sim_mat=None, pprm_supervision='repeat',
                spatial_gate=False, target_space='level', quantiles=False,
                attn_fix=False, attn_exp=None, renewal=False, renewal_lag=None,
                gi_fix=None, n_quantiles=None, level_cap=None,
                model_name=MODEL_NAME):
    """Return the canonical run token.

    Mirrors ``src/train.py`` exactly for every historical argument. ``sim_mat``
    contributes a segment only when explicitly supplied -- a dataset default
    leaves the token unchanged, which is why every artefact on disk reads
    ``sim_mat='default'`` in its CSV row but carries no ``sim`` segment.
    """
    if use_adj_prior is None:          # legacy pre-train.py runner
        adj_tag = ''
    else:
        adj_tag = 'with_adj' if use_adj_prior else 'no_adj'
    sim_tag = f".{sim_mat}" if sim_mat else ""
    variant = build_variant_tag(
        pprm_supervision=pprm_supervision, spatial_gate=spatial_gate,
        target_space=target_space, quantiles=quantiles, attn_fix=attn_fix,
        attn_exp=attn_exp, renewal=renewal, renewal_lag=renewal_lag,
        gi_fix=gi_fix, n_quantiles=n_quantiles, level_cap=level_cap)

    head = (f"{model_name}.{dataset}.w-{window}.h-{horizon}."
            f"{ablation}.seed-{seed}")
    if adj_tag:
        head += f".{adj_tag}"
    return head + sim_tag + variant


def build_variant_tag(pprm_supervision='repeat', spatial_gate=False,
                      target_space='level', quantiles=False, attn_fix=False,
                      attn_exp=None, renewal=False, renewal_lag=None,
                      gi_fix=None, n_quantiles=None, level_cap=None):
    """Return just the variant suffix of a token.

    ``save_metrics`` keys its dedup mask on the ``model`` column, which is
    ``"MSAGAT-Net" + variant_tag``. Sharing this function with
    :func:`build_token` keeps the CSV key and the filename from drifting
    apart, which is how a 7-quantile run and a 23-quantile run came to
    collide on both.
    """
    variant = ""
    if pprm_supervision != 'repeat':
        variant += f".pprm-{pprm_supervision}"
    if spatial_gate:
        variant += ".sgate"
    if target_space != 'level':
        variant += f".{target_space}"
    if quantiles:
        variant += ".quant"
    if attn_fix:
        variant += ".attnfix"
    if attn_exp:
        variant += ".exp-" + attn_exp.replace(',', '-')
    if renewal:
        variant += f".renewal{renewal_lag or ''}"
    if gi_fix:
        variant += f".gifix{gi_fix[0]:g}-{gi_fix[1]:g}"

    # New segments: no-ops at the historical defaults.
    if quantiles and n_quantiles not in (None, DEFAULT_N_QUANTILES):
        variant += f".q{n_quantiles}"
    if (target_space != 'level' and level_cap is not None
            and level_cap != DEFAULT_LEVEL_CAP):
        variant += f".cap{level_cap:g}"
    return variant


_HEAD = re.compile(
    r'^(?P<model_name>[^.]+)\.'
    r'(?P<dataset>[^.]+)\.'
    r'w-(?P<window>\d+)\.'
    r'h-(?P<horizon>\d+)\.'
    r'(?P<ablation>[^.]+)\.'
    r'seed-(?P<seed>\d+)'
    r'(?:\.(?P<adj>with_adj|no_adj))?'
    r'(?P<rest>.*)$'
)

# Consumed left to right in exactly the order build_token emits them. Segments
# are NOT found by splitting on "." -- gifix and cap carry decimal points, and
# splitting was the bug the round-trip test caught.
_SEGMENTS = [
    ('pprm_supervision', re.compile(r'\.pprm-([A-Za-z0-9_]+)')),
    ('spatial_gate', re.compile(r'\.sgate')),
    ('target_space', re.compile(r'\.(loggrowth)')),
    ('quantiles', re.compile(r'\.quant(?![A-Za-z0-9])')),
    ('attn_fix', re.compile(r'\.attnfix')),
    ('attn_exp', re.compile(r'\.exp-([^.]+)')),
    ('renewal', re.compile(r'\.renewal(\d*)')),
    ('gi_fix', re.compile(r'\.gifix(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)')),
    ('n_quantiles', re.compile(r'\.q(\d+)')),
    ('level_cap', re.compile(r'\.cap(\d+(?:\.\d+)?)')),
]

_FLAGS = ('spatial_gate', 'quantiles', 'attn_fix')

# A sim_mat segment, when present, comes first and matches none of the above.
_SIM = re.compile(r'\.([^.]+)')


def parse_token(token):
    """Invert :func:`build_token`.

    Returns a dict of keyword arguments that, passed back to ``build_token``,
    reproduces the input string. Raises ``ValueError`` on anything that is not
    a well-formed token, so a malformed filename fails loudly rather than
    being silently misattributed to another configuration.

    Legacy tokens from the pre-``train.py`` runner carry no adjacency segment
    (``...seed-5``); those round-trip with ``use_adj_prior=None``.
    """
    token = os.path.basename(token)
    for suffix in ('.npz', '.pt', '.npy', '.json'):
        if token.endswith(suffix):
            token = token[:-len(suffix)]

    m = _HEAD.match(token)
    if m is None:
        raise ValueError(f'not a run token: {token!r}')

    adj = m.group('adj')
    spec = {
        'model_name': m.group('model_name'),
        'dataset': m.group('dataset'),
        'window': int(m.group('window')),
        'horizon': int(m.group('horizon')),
        'ablation': m.group('ablation'),
        'seed': int(m.group('seed')),
        'use_adj_prior': None if adj is None else (adj == 'with_adj'),
        'sim_mat': None,
        'pprm_supervision': 'repeat',
        'spatial_gate': False,
        'target_space': 'level',
        'quantiles': False,
        'attn_fix': False,
        'attn_exp': None,
        'renewal': False,
        'renewal_lag': None,
        'gi_fix': None,
        'n_quantiles': None,
        'level_cap': None,
    }

    rest = m.group('rest')
    pos = 0

    if pos < len(rest):
        # sim_mat is the first segment only if no known pattern starts here.
        if not any(pat.match(rest, pos) for _, pat in _SEGMENTS):
            hit = _SIM.match(rest, pos)
            if hit is None:
                raise ValueError(f'unparsable segment at {rest[pos:]!r} in {token!r}')
            spec['sim_mat'] = hit.group(1)
            pos = hit.end()

    for name, pat in _SEGMENTS:
        hit = pat.match(rest, pos)
        if hit is None:
            continue
        if name in _FLAGS:
            spec[name] = True
        elif name == 'target_space':
            spec['target_space'] = hit.group(1)
        elif name == 'renewal':
            spec['renewal'] = True
            spec['renewal_lag'] = int(hit.group(1)) if hit.group(1) else None
        elif name == 'gi_fix':
            spec['gi_fix'] = (float(hit.group(1)), float(hit.group(2)))
        elif name == 'n_quantiles':
            spec['n_quantiles'] = int(hit.group(1))
        elif name == 'level_cap':
            spec['level_cap'] = float(hit.group(1))
        else:
            spec[name] = hit.group(1)
        pos = hit.end()

    if pos != len(rest):
        raise ValueError(f'unrecognised token segment {rest[pos:]!r} in {token!r}')

    return spec


def _base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def npz_path(token, dataset=None, base=None):
    """Path of the prediction archive for ``token``."""
    base = base or _base_dir()
    dataset = dataset or parse_token(token)['dataset']
    return os.path.join(base, 'report', 'predictions', dataset, token + '.npz')


def ckpt_path(token, save_dir='save_all', base=None):
    """Path of the checkpoint for ``token`` in ``save_dir``."""
    base = base or _base_dir()
    return os.path.join(base, save_dir, token + '.pt')


def attention_path(token, base=None):
    """Path of the persisted attention matrix for ``token``."""
    base = base or _base_dir()
    return os.path.join(base, 'report', 'attention', token + '.npy')


def manifest_path(token, dataset=None, base=None):
    """Path of the run manifest for ``token``."""
    base = base or _base_dir()
    dataset = dataset or parse_token(token)['dataset']
    return os.path.join(base, 'report', 'manifests', dataset, token + '.json')
