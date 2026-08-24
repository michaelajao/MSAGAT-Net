"""Merge-on-write for result CSVs.

``conformal.py`` and ``prob_eval.py`` both ended with ``df.to_csv(OUT_CSV)``,
which replaces the entire file. A filtered run -- ``--dataset nhs_timeseries``,
or ``--split val`` -- therefore silently destroyed every row it did not
recompute: 450 conformal rows collapse to about four. Nothing warned, and the
result looked like a completed file.

``merge_rows`` writes the same way ``utils.save_metrics`` already does for
``all_results.csv``: read what is there, drop only the rows the new batch
supersedes, concatenate, write once. Passing ``--replace-all`` remains
available for a deliberate full regeneration.
"""

import os

import pandas as pd

__all__ = ['merge_rows']


def merge_rows(path, rows, keys, replace_all=False, verbose=True):
    """Write ``rows`` to ``path``, superseding existing rows on ``keys``.

    ``keys`` is the list of columns that identify a result uniquely. Existing
    rows whose key tuple appears in the new batch are replaced; every other
    row is kept. Returns the written DataFrame.
    """
    new = pd.DataFrame(rows)
    if new.empty:
        if verbose:
            print(f'nothing to write to {path}')
        return new

    missing = [k for k in keys if k not in new.columns]
    if missing:
        raise KeyError(f'new rows are missing key columns {missing}')

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    if replace_all or not os.path.exists(path):
        new.to_csv(path, index=False)
        if verbose:
            action = 'replaced' if replace_all else 'created'
            print(f'{action} {path} with {len(new)} rows')
        return new

    old = pd.read_csv(path)
    if not all(k in old.columns for k in keys):
        # A schema change means the old file cannot be keyed against; keep it
        # rather than silently dropping it.
        backup = path + '.superseded'
        old.to_csv(backup, index=False)
        new.to_csv(path, index=False)
        if verbose:
            print(f'schema changed; previous file kept at {backup}')
            print(f'wrote {path} with {len(new)} rows')
        return new

    incoming = set(map(tuple, new[keys].astype(str).values))
    mask = old[keys].astype(str).apply(tuple, axis=1).isin(incoming)
    kept = old[~mask]
    merged = pd.concat([kept, new], ignore_index=True)
    merged.to_csv(path, index=False)
    if verbose:
        print(f'wrote {path}: {len(kept)} kept + {len(new)} new '
              f'({int(mask.sum())} superseded) = {len(merged)} rows')
    return merged
