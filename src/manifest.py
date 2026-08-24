"""Per-run manifests: what produced a number, recorded next to the number.

Before this, a run's only identity was its filename token, and the token
omits four things that change the results: the growth level cap, the quantile
count, whether the run was a fresh train or an ``--eval_only`` re-score, and
every entry of ``TRAIN_DEFAULTS``. Nothing recorded the git commit, and
``report/`` was gitignored, so on 14 August a re-scoring pass silently
overwrote 105 prediction archives and their CSV rows in place with no trace
beyond a log file that was itself untracked.

A manifest is written beside every prediction archive:

    report/manifests/{dataset}/{run_token}.json

It is additive -- no existing artefact changes -- and it is small enough to
keep under version control, so a claim in a paper can be checked against the
exact code and configuration that produced it even though the checkpoint and
the npz stay out of the repository. The SHA-256 of each artefact is recorded
so a file can be matched to its manifest after the fact.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys

__all__ = ['git_state', 'sha256', 'write_manifest']

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_git(args):
    """Return stripped stdout of a git command, or None if git is unusable."""
    proc = subprocess.run(['git'] + args, cwd=_BASE, capture_output=True,
                          text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_state():
    """Commit hash, branch, and whether tracked source files are dirty.

    ``dirty`` considers only ``src/``: a modified manuscript or a new result
    CSV does not change what the code did, but an uncommitted source edit
    means the recorded commit does not fully describe the run.
    """
    commit = _run_git(['rev-parse', 'HEAD'])
    branch = _run_git(['rev-parse', '--abbrev-ref', 'HEAD'])
    status = _run_git(['status', '--porcelain', '--', 'src'])
    return {
        'commit': commit,
        'branch': branch,
        'src_dirty': bool(status) if status is not None else None,
        'src_dirty_files': status.splitlines() if status else [],
    }


def sha256(path, chunk=1 << 20):
    """SHA-256 of a file, or None if it does not exist."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _jsonable(value):
    """Coerce argparse/numpy values into something json can write."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


def write_manifest(path, run_token, dataset, config, constants, artefacts,
                   metrics, training=None, data=None, mode='train',
                   started_utc=None, finished_utc=None, wall_seconds=None):
    """Write one run manifest and return the path.

    ``artefacts`` maps a role ("npz", "checkpoint", "attention", ...) to a
    filesystem path; each is hashed here so the manifest can be matched to
    the file later even if the file moves.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    hashed = {}
    for role, target in (artefacts or {}).items():
        hashed[role] = {
            'path': (os.path.relpath(target, _BASE).replace('\\', '/')
                     if target else None),
            'sha256': sha256(target),
        }

    record = {
        'schema_version': 1,
        'run_token': run_token,
        'dataset': dataset,
        'mode': mode,
        'started_utc': started_utc,
        'finished_utc': finished_utc,
        'wall_seconds': wall_seconds,
        'git': git_state(),
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'argv': sys.argv,
        },
        'config': _jsonable(config),
        'constants': _jsonable(constants),
        'training': _jsonable(training or {}),
        'data': _jsonable(data or {}),
        'artefacts': hashed,
        'metrics': _jsonable(metrics),
    }

    try:
        import torch
        record['environment']['torch'] = torch.__version__
        record['environment']['cuda'] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass

    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=2, sort_keys=False)
        fh.write('\n')
    return path
