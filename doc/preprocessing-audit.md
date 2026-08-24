# Preprocessing / smoothing leakage audit

Requested by AIIM Reviewer #1 point 9 and independently by the April 2026
technical review (findings 4 and 5). Ledger section 3 marked it blocking.
Completed 19 August 2026. **Verdict: no future-information leakage found.**
Two conventions must be *stated* in the paper rather than left implicit.

## Questions and answers

**Q1. Is smoothing applied before or after the chronological split?**
Before — but causally. No runtime smoothing exists in any of the three repos
(grep over `src/` of MSAGAT-Net, colagnn, EpiGNN: zero hits outside a comment
in `conformal.py`). LTLA and NHS files were smoothed offline at dataset-creation
time with a **trailing (causal) 7-day mean** — established earlier from the
denominator signature (1, 2, …, 7, 7, 7 at the series head). A trailing mean at
position t uses only observations t−6…t, so **no value in any file contains
future information**, and smoothing-before-splitting cannot leak the future
into targets. The dangerous variant would be a centered mean (t−3…t+3); the
signature rules it out. Australia is raw (unsmoothed) — the manuscript text
claiming all three daily datasets are smoothed was corrected earlier; weekly
ILI datasets (japan, region785, state360) are not smoothed.

**Q2. Does normalization leak across splits?**
No. Min–max parameters are computed from the training range only
([src/data.py:141-159](../src/data.py): `_compute_normalization` batchifies
`train_set` with `useraw=True` and takes max/min of train inputs+targets), then
applied to the whole series. colagnn `_pre_train` mirrors this exactly.

**Q3. Is the split chronological, and do sliding windows cross boundaries?**
The split is strictly chronological: samples are indexed by target time `idx`
with `train_set = range(P+h−1, train_end)`, `valid_set = [train_end, val_end)`,
`test_set = [val_end, n)`; shuffling happens only within training batches.
**Input windows do cross split boundaries**: a test sample at `idx = val_end`
has input window `[idx−h+1−P, idx−h+1)`, which reaches back into validation
(and for small val splits, train). This is the standard rolling-origin
convention of the Cola-GNN benchmark family, it exposes only *past*
observations (causal, exactly what a deployed forecaster would see), and it is
identical for every model compared. It is not leakage, but the paper should
state it explicitly in the evaluation-protocol paragraph.

**Q4. Do baselines receive identical inputs?**
Yes, byte-identical: md5 checksums of all six dataset files match across
MSAGAT-Net, colagnn, and EpiGNN repos (verified 2026-08-19, all IDENTICAL).
Split index ranges also match line-for-line across repos (colagnn
`src/data.py:53-55` vs MSAGAT-Net `src/data.py:143-145`), which is what makes
the persisted per-timestep predictions alignable sample-for-sample for the
DM tests.

## For the manuscript

1. Evaluation-protocol paragraph: state the trailing/causal smoothing (LTLA,
   NHS only; Australia raw; weekly datasets unsmoothed), train-only
   normalization, chronological 60/20/20 split, and the rolling-origin
   convention that input windows may span split boundaries while targets never
   precede the split start.
2. Metrics on LTLA/NHS are on the smoothed scale — keep the existing
   limitation sentence.
