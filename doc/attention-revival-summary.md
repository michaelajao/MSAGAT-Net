# EAGAM attention-revival campaign — results

Deliverable requested by `program.md`. Campaign run 19 August 2026.
Per-config numbers: `report/results/attn_revival_ledger.csv`;
per-cell rows: `report/results/attn_revival_runs.csv`.

## Outcome

**The campaign succeeded, but with a smaller and less novel result than the
programme hoped for.** EAGAM can be made to carry genuine signal, and the fix
that does it is two lines. Every one of the nine elaborations of that fix was
worse. A 5-seed, full-grid confirmation is running; until it lands, everything
below is validation-only, seed 42, on the five-cell proxy grid.

**What the fix is — and is not.** `nodecay,regpre` is an improved *training
recipe*, not a new architecture. The model is byte-identical: same modules,
same parameter count. `nodecay` (excluding parameters from weight decay) is
standard practice; `regpre` repairs a penalty that never worked. No reviewer
should be asked to read this as architectural novelty. The publishable content
is the **diagnosis and the documented failure mode**, not the patch.

## The bar was wrong, and the correction matters

`program.md` specified the comparison point as the ablated model with EAGAM
removed, "which is stronger". In v2 space (log-growth + quantiles, validation)
it is **worse in 5/5 proxy cells, mean +27.1%** — NHS h3 +55.0%, h7 +31.0%,
h14 +16.0%, Japan h3 +22.3%, h5 +11.4%.

That does not contradict the earlier "-7.95% when removed" ablation, which was
measured in level space on test. Both hold; they are different settings. The
reconciliation is mechanical: `IdentitySpatialModule` strips out *all* spatial
mixing, whereas uniform attention times values is **unweighted spatial mean
pooling**, added residually. So inert attention is **non-selective, not
useless** — the aggregation earns its place even when the selectivity is
absent. The operative bar is therefore **frozen v2 per cell**, the harder
target, and the campaign was run against it.

## Result table

Bar = frozen v2 per cell (nhs h3 2.8285, h7 7.2757, h14 19.8442; japan h3
627.7834, h5 604.6056). Gate = mean attention row entropy <= 0.98 **and**
learned-term share of pre-softmax score variance >= 0.10, applied *before* RMSE.

| # | config | direction | gate (ent / uv) | vs bar | vs exp-1 | verdict |
|---|---|---|---|---|---|---|
| 1 | **`nodecay,regpre`** | 1a+1b | PASS (0.854 / 0.780) | **4/5, −6.9%** | — | **KEPT** |
| 3R | `+init10` | 3 | PASS (0.736 / 0.854) | 2/5, −4.9% | +2.3% | discard |
| 11 | `+regent` | 10 | PASS (0.742 / 0.883) | — | +2.8% | discard |
| 4 | `+lrx10` | 4 | PASS (0.713 / 0.895) | 5/5, −3.4% | +4.6% | discard |
| 10 | `+adjstd,lrx10` | combo | PASS (0.688 / 0.848) | 4/5, −0.9% | +6.9% | discard |
| 5 | `+adjstd` | 7 | PASS (0.802 / 0.618) | 3/5, −1.8% | +7.0% | discard |
| 2R | `+temp` | 2 | PASS (0.851 / 0.639) | 3/5, +0.9% | +9.1% | discard |
| 6 | `+scorenorm` | 8 | PASS (0.773 / 0.662) | 2/5, +2.0% | +9.8% | discard |
| 9 | `+rank16` | 6 | PASS (0.915 / 0.663) | 2/5, +8.4% | +17.1% | discard |
| 8 | `+rank2` | 6 | PASS (0.960 / 0.526) | 2/5, +13.4% | +22.2% | discard |
| 7 | `+multgate` | 9 | **FAIL** (0.995 / n/a) | — | — | **gate discard** |

**Configuration count for multiple-comparisons reporting: 11 configs trained,
of which 2 (`temp`, `init10`) were invalidated by a harness bug and re-run —
13 training campaigns total, 11 distinct configurations scored.** The
configuration was selected entirely on validation; the test split is touched
once, by the 5-seed confirmation.

## Which gate condition failed, and why the gate earned its place

Only one config failed the gate, and it failed on **entropy**: `multgate`
(direction 9) reached mean row entropy **0.9952** — attention back to uniform —
while posting a mean RMSE of 234.51 against exp-1's 233.30, essentially tied,
plus the campaign's best Japan h=5 (532.91). **On an RMSE-only protocol it
would have been recorded as a success.** This is precisely the failure
`program.md` anticipated: the cheapest way to lower RMSE is to suppress
attention back toward the mean-pooling solution that already works.

Mechanism: `scores x sigmoid(U@V)` has a degenerate optimum at `U@V -> 0`,
where the gate becomes constant and the softmax flattens — and the `regpre` L1
actively pushes toward it. The two changes fight, and the shortcut wins.

*Caveat recorded with that row:* `uv_share` is computed as
`var(content + bias)`, which assumes the additive formulation. Under
`multgate` the model computes `content * sigmoid(bias)`, so that column is not
meaningful there. The discard rests on entropy, measured directly from the
realised attention.

## What the campaign established beyond the winner

- **Interior optimum in selectivity.** Accuracy degrades on *both* sides of
  exp-1's uv_share ≈ 0.78: less selective (`temp`, 0.639) is +9.1%, more
  selective (`init10` 0.854, `lrx10` 0.895) is +2.3% and +4.6%. Sharper
  attention is not better attention, so the gate is a **floor, not an
  objective**.
- **Whole-vector rescaling fails.** Both changes that rescale the entire score
  vector — learnable temperature and score normalisation — are among the worst
  (+9.1%, +9.8%). What matters is *relative* structure within a row.
- **Rank 8 is already right.** The sweep is symmetric: rank 2 is +22.2%
  (Japan collapses to 817/866 — rank 2 cannot express 47-node structure) and
  rank 16 is +17.1%. Capacity is not the binding constraint. Direction 6 closed.
- **Complementarity does not compose.** `adjstd` alone fixes exp-1's only
  losing cell (Japan h=5: 635.46 -> 552.93). Combined with `lrx10`, that same
  cell went to **694.12 — worse than either change alone**.
- **The static prior is silenced by graph size.** Softmax is row-shift-
  invariant, so only *within-row* variation can influence attention. Row-
  normalising a dense graph destroys that variation: measured within-row sd of
  the normalised prior is **0.005 on 372-node LTLA versus 0.197 on 7-node
  NHS**, a 38x gap driven purely by density. Geography stops mattering exactly
  where it should matter most. (Motivates the density-invariant standardisation
  now used by the renewal decoder.)

## Method notes — two silent bugs worth reporting

Both would have invalidated conclusions without changing any visible output.

1. **`_init_weights()` overwrote experiment parameters.**
   `MSAGATNet_Ablation._init_weights()` re-applies `xavier_uniform_` to every
   parameter of dim >= 2 not in `_PRESERVE_PARAMS` — which included
   `graph_attention.u`/`v` — and `uniform_(-1, 1)` to 1-D ones, which included
   `log_attn_temp`. So the `init10` scaling was wiped (exp-3 came back
   *bit-identical* to exp-1, which is how it was caught) and the temperature
   never started at its intended 1.0. Fixed by adding `log_attn_temp` to
   `_PRESERVE_PARAMS` and moving the scaling into `apply_init_scale()`, called
   after the global init pass.
2. **Token parsed before it existed.** `rankN` was read before `self.attn_exp`
   was assigned, raising on construction.

**Practice adopted:** verify a new token actually changes behaviour at
construction — build the model twice and compare attention entropy — before
trusting any run. A silently-null experiment reports as a clean negative.

**Scoring practice:** score per cell, never on the raw mean. `regent`'s mean
RMSE (228.74) is *lower* than exp-1's (233.30) and looks like a win; per cell
it loses 3/5 and averages +2.8% worse. Japan's scale is ~80x NHS's and
dominates any unweighted mean.

## For the paper

Frame this as a **documented failure mode with a mechanism**, supported by
three independent lines of evidence that agree:

1. **Analytical** — the sparsity penalty is provably inert. `L = lambda*||A||_1`
   on a row-wise softmax output is exactly `lambda/N`: the gradient w.r.t. the
   attention is identically zero, and the gradient on the *learnable* lambda is
   positive, so lambda decays itself away. Doubly dead.
2. **Diagnostic** — with nothing opposing it, weight decay drove `u`,`v` to
   **~1e-36**; attention row entropy reached **1.0000** on LTLA (minimum
   0.9998 across every row, head and test sample); the only score term with any
   spread was the *static* adjacency prior.
3. **Behavioural** — the module degenerates to mean pooling: removing it is a
   no-op in level space on LTLA (−0.1% to +1.4%) yet costs +27.1% in v2 space,
   because what it contributes is aggregation, not selection.

The fix restores the module (learned-term variance share ~0.00 -> 0.78,
entropy 0.994 -> 0.854) and improves accuracy, and **nine attempts to improve
on it all failed** — which is itself the strongest evidence that the diagnosis,
not the patch, is the contribution.

## What this does not do

It does not deliver architectural novelty, and it will not beat SOTA. The
novelty push is the renewal-equation decoder (Track B); see
`doc/adversarial-priority-check.md` for what claims survive there.
