> **CLOSED (20 Aug 2026).** Both programmes this file defined are complete.
> Attention revival: winner `nodecay,regpre`, all nine elaborations worse, but
> the fix did not generalise on the 5-seed test confirmation — see
> `doc/attention-revival-summary.md`. Renewal: 45 comparisons, 0 accuracy wins,
> but the interpretability results survived multi-seed confirmation and became
> Paper A — see `doc/renewal-net-design.md` and
> `doc/paper-renewal-interpretability.md`. Kept for the record; do not re-run.

# Research programme: can a learned renewal equation carry the model?

## Status of the previous programme

The EAGAM attention-revival programme that occupied this file is **complete**.
Result: EAGAM can be made to carry signal, by a two-line fix
(`nodecay,regpre`) that beats the frozen-v2 bar in 4/5 proxy cells (−6.9%,
validation) and lifts the learned-term share of attention score variance from
~0.00 to 0.78. All nine elaborations of that fix were worse. Full write-up,
including the configuration count and two method-note bugs, is in
`doc/attention-revival-summary.md`. Do not re-run it.

Two corrections it produced, which this programme inherits:

- The no_agam ablation is **not** a stronger bar. In v2 space it is worse in
  5/5 cells (+27.1%). Inert attention is *non-selective, not useless* —
  uniform attention times values is unweighted spatial mean pooling.
- **Score per cell, never on the raw mean.** Japan's scale is ~80x NHS's and
  dominates any unweighted average; one config's mean RMSE looked better than
  the winner's while losing 3 of 5 cells.

That programme delivered a diagnosis, not novelty. The architecture is
byte-identical, `nodecay` is standard practice, `regpre` is a bug fix. This
programme is the novelty push.

## Background for the agent

Two things this project has already established turn out to be the same object.

**Empirical.** The largest measured effect anywhere in the work is the
log-growth target space, `g = log((y_t + 1)/(y_anchor + 1))`. It helps in
**15 of 23 baseline cells**, up to −51.9%, and it transfers across
architectures — so it is a design principle, not a trick. But it is currently
justified only by results.

**Mechanistic.** The renewal equation is
`I(t) = R_t * SUM_tau w_tau * I(t-tau)`, with `w_tau` the generation-interval
distribution (Fraser 2007; Cori et al. 2013, EpiEstim). Taking logs,
**predicting log-growth is predicting `log R_t`**, up to the convolution term.

So the network should predict the reproduction number, and a learned
generation-interval kernel should supply the rest:

```
I_i(t+h) = R_i(t) * SUM_tau alpha_tau * SUM_j w_ij * I_j(t-tau)
           ^^^^^^^^          ^^^^^^^^^^          ^^^^
           backbone          learned GI          spatial coupling
```

This is implemented and training: `--renewal [--renewal_lag N]`, decoder in
`MSAGATNet_Ablation.forward` plus `_renewal_coupling()`. It reuses the existing
backbone, so renewal-vs-direct is a controlled decoder comparison. Design notes
and the claim wording that survives review: `doc/renewal-net-design.md`.

**CORRECTION, read before trusting any kernel figure.** Because the input
window ends at `dat[idx-h]`, kernel index `tau` is a delay of **`h + tau`** from
the target — there is no zero-delay term, and Cori's `w_0 = 0` convention does
not apply here. Earlier "mean lag ~3.5 days" figures were means of `tau`, not of
delay, and averaged across daily and weekly cells. The generation-interval
reading does **not** hold for h > 1: the infections generating the target occur
in the unobserved gap between `idx-h` and `idx`, so a single convolution over
older history is a stale-delay kernel, not the renewal equation. A faithful
version must iterate the equation forward h times. See the CORRECTION section in
`doc/renewal-net-design.md`.

Measured so far (proxy grid, validation, seed 42, vs the exp-1 bar): R1
(tau>=1) **loses 5/5, +23.4%**; R2 (tau>=0, the ablation) loses 5/5, +12.5% —
i.e. the "principled" exclusion is the worse of the two, and the degeneracy it
guarded against never occurred (alpha_first 0.2877, entropy 0.8165).

The question this programme answers is narrow and has two parts, and **both
must hold**: does the learned kernel recover an epidemiologically plausible
generation interval, **and** does the renewal decoder beat the direct decoder?
Either alone is not enough. A plausible kernel that loses on accuracy is a
curiosity; an accuracy win with a degenerate kernel is the renewal structure
doing nothing.

## The bar

The comparison point is **exp-1 (`nodecay,regpre`) per cell**, on validation,
on the proxy grid — the strongest current model, with the attention fix already
applied. Renewal configs must be run with the fix on, so the only difference is
the decoder.

Proxy cells and bar values (validation RMSE, seed 42):

| cell | bar (exp-1) |
|---|---|
| nhs_timeseries h3 | 2.8082 |
| nhs_timeseries h7 | 7.1788 |
| nhs_timeseries h14 | 16.3110 |
| japan h3 | 504.7393 |
| japan h5 | 635.4625 |

Report frozen v2 alongside (2.8285 / 7.2757 / 19.8442 / 627.7834 / 604.6056) so
the decoder change and the attention fix stay separable.

## Scoring

Primary metric: **validation RMSE, compared per cell** against the bar. Report
the count of cells beaten and the mean of per-cell percentage changes. Never
rank on the raw mean across cells.

**The degeneracy is designed out, not merely detected.** The previous
programme relied on a gate to *catch* a collapsed module. That is weaker than an
architecture that cannot collapse, so the decoder now excludes `tau=0`: the
kernel spans `tau = 1..L`. This is the standard epidemiological convention
(Cori et al. 2013 set `w_0 = 0` — in discrete time nobody is infected at zero
delay) and it forces `alpha` to be a genuine **delay** distribution.

Be precise about what this does and does not buy, because the first version of
this section overstated it. With `tau=0` allowed, `alpha` can collapse to a
delta there and `Lambda` becomes `SUM_j w_ij I_j(t)` — a purely spatial,
zero-delay aggregate with no temporal structure, which guts the
generation-interval reading while leaving RMSE untouched. It degenerates all the
way to `offset == 0`, reproducing the direct log-growth decoder exactly, only if
the spatial coupling *also* approaches the identity; row-softmax makes that
unlikely but not impossible. Excluding `tau=0` removes the first and much more
reachable failure. `renewal_lag0` restores the old behaviour so the collapse can
be **demonstrated as an ablation** rather than asserted — run it once, early,
and report it.

**Gate — applied before the metric is even considered.** A run is discarded,
whatever its RMSE, unless the learned kernel is non-degenerate:

1. **Mass at the first lag at most 0.60**, so the kernel is not a
   near-delta that reduces the convolution to a single lagged observation.
2. **Mean lag at least 1.5 steps.**
3. **Kernel entropy at least 0.5** of the uniform maximum over its support.

Log all three for every run, kept or discarded, plus the full `alpha` vector and
the mean/spread of predicted `R`. The distribution of near-misses is itself
informative, and the `alpha` vectors are needed for the plausibility figure.

**Plausibility is reported, not gated.** State the mean lag against literature
generation intervals (COVID ~3–5 days; influenza ~2–4 days, and note the weekly
datasets are in weeks, not days). Do not tune toward the literature value —
that would make the recovery circular, which is the whole point of the claim.

## Directions to explore

Roughly in order of expected value. One change at a time; combine only after
individual effects are known.

1. **`renewal_lag` sweep: 7, 14, 21.** Cheapest, and the current default may
   simply be wrong for weekly versus daily data. Weekly series almost certainly
   want a much shorter kernel.
2. **Residual variant.** Let the backbone predict a correction alongside
   `log R` rather than `log R` alone. Fixing the convolution entirely to data
   may over-constrain the model, which is the most likely cause of the first
   run's accuracy loss. This is the highest-value structural change.
3. **Initialise `alpha` from a literature generation interval** (discretised
   gamma) instead of uniform. Mark clearly: this weakens the "recovered from
   data" claim, so it must be reported separately from uniform-init runs, and
   the uniform-init result is the one that supports the interpretability claim.
4. **Constrain `R` to a plausible range**, e.g. a scaled sigmoid giving
   `R in [0, 10]`, instead of an unconstrained `log R`.
5. **Per-node `alpha`** instead of one shared kernel — does the generation
   interval differ by region, or is a single kernel sufficient?
6. **Decouple the spatial coupling** used by the renewal convolution from the
   attention module's, so the convolution can use geography while attention
   does something else.
7. **Multiple kernels** (a small mixture over `alpha`), for pathogens or waves
   with different generation intervals.
8. **Weakly supervise `alpha`** with a KL penalty toward a literature GI. Try
   this **last**: it forces plausibility by construction rather than
   discovering it, so a win here is much weaker evidence. Mark any such run.

## The decisive experiment

Required before any claim is made, per `doc/adversarial-priority-check.md`
recommendation 3. Not optional, and not a direction to be reached only if time
allows:

- learned `alpha_tau` versus a **fixed** literature generation interval,
- versus an **EpiEstim-style renewal baseline** (fixed GI, `R_t` estimated
  classically, no network),
- with the same corrected protocol, 5 seeds, and DM tests.

The claim survives only if the learned kernel both recovers a plausible shape
**and** improves accuracy over the fixed kernel. If it recovers the shape but
does not improve accuracy, say so — that is still a publishable interpretability
result, but it must not be dressed as an accuracy contribution.

## Hard constraints — never violate

- **Do not modify any evaluation code.** Forecasts are scored at lead time *h*
  only. A pooled *h*…2*h*−1 scoring bug was found and fixed in this repository;
  reintroducing it would silently inflate every result.
- **Do not touch the test split.** All ratcheting is on validation. The test
  split is contacted once, at the end, for the confirmation run.
- **Do not modify data loading, splitting, or the smoothing pipeline.** The
  leakage audit is complete and clean (`doc/preprocessing-audit.md`); changing
  these would invalidate it.
- **Do not modify the Diebold-Mariano harness or any significance-testing code.**
- **Keep the validation-selected level cap** on log-growth inversion (3x
  per-node training max).
- **Do not change the target space.** Log-growth targets stay — they are now the
  mechanism, not just a setting.
- **Keep the attention fix (`nodecay,regpre`) on** in every renewal run, so the
  decoder is the only thing varying.
- Change one thing per experiment. Revert cleanly on discard.

## Verification practice — learned the hard way

Two silent bugs invalidated runs in the previous programme; both reported clean
results while doing nothing.

**Before trusting any new token or flag, verify it changes behaviour at
construction.** Build the model twice, with and without, and compare a concrete
quantity — attention entropy, the `alpha` vector, a parameter norm. A
silently-null experiment is indistinguishable from a genuine negative result,
and `MSAGATNet_Ablation._init_weights()` in particular re-initialises every
parameter of dim >= 2 not in `_PRESERVE_PARAMS` (note `log_alpha` is already
listed there, so its uniform init survives — keep it that way).

## Logging

For every run, kept or discarded: the configuration diff, validation RMSE per
cell, the three gate quantities, the full `alpha` vector, mean and spread of
predicted `R`, and the norms of `u`/`v`. The count of total configurations tried
must be preserved — it is needed for multiple-comparisons reporting, and the
previous programme's count (11 scored, 13 trained) is already in the paper's
obligations.

## Stopping

Stop after 60 runs, or after 20 consecutive runs with no kept improvement,
whichever comes first. Tighter than the previous programme because the decisive
experiment matters more than the search, and because a null result here is
reached faster: if the residual variant (direction 2) and the lag sweep both
fail, the renewal decoder is unlikely to be rescued by directions 5–8.

If the campaign ends without a configuration that passes the gate *and* beats
the bar, write a summary of what was tried and which gate condition failed most
often. **That is a genuinely publishable outcome**: it would show that the
renewal reparameterisation recovers interpretable epidemiological structure
without improving forecast accuracy, which is a real and citable finding about
the limits of mechanistic inductive bias in neural epidemic forecasting.

## What happens next (not for the agent)

If the renewal line succeeds on both counts, it becomes the paper's headline
contribution, positioned per `doc/adversarial-priority-check.md`: a
differentiable, spatially-coupled renewal equation with an end-to-end learned
generation interval, benchmarked against both epidemiological (EpiEstim/EpiNow2)
and deep-learning (PDFormer, EpiGNN, Cola-GNN, MepoGNN) baselines under the
corrected protocol. The "first to model propagation delay" framing is dead and
must not reappear.

If it fails, the paper is the evaluation paper: corrected protocol (E1),
target-space design as a transferable principle (E2), first calibrated
probabilistic forecaster on these datasets (E6), and the attention failure mode
(E3/E7) as a documented negative result. That paper is safe, defensible, and
mostly already written.
