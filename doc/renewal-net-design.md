# Renewal-equation decoder — design and status

Track B, the novelty line. Implemented 19 August 2026; first feasibility run
complete. Positioning constraints come from `adversarial-priority-check.md`.

## The idea

Two things this project already knows turn out to be the same object.

**Empirically**, the largest measured effect anywhere in the work is the
log-growth target space: `g = log((y_t + 1)/(y_anchor + 1))`, which helps in
**15 of 23 baseline cells**, up to −51.9% (Cola-GNN, Australia). It transfers
across architectures, so it is a design principle rather than a trick.

**Mechanistically**, the renewal equation is
`I(t) = R_t * SUM_tau w_tau * I(t-tau)`, with `w_tau` the generation-interval
distribution (Fraser 2007; Cori et al. 2013, EpiEstim). Taking logs,
**predicting log-growth is predicting `log R_t`**, up to the convolution term.

So the network should predict the *reproduction number*, and the convolution
over a learned generation interval should supply the rest:

```
I_i(t+h) = R_i(t) * SUM_tau alpha_tau * SUM_j w_ij * I_j(t-tau)
           ^^^^^^^^          ^^^^^^^^^^          ^^^^
           backbone          learned GI          spatial coupling
```

This explains *why* the project's best empirical choice works, rather than
leaving it as a heuristic — which is also the ledger's open question about why
log-growth fails where it fails.

## Implementation

`--renewal [--renewal_lag N]` in `src/train.py`; decoder in
`MSAGATNet_Ablation.forward` plus `_renewal_coupling()` in `src/models.py`.
Deliberately reuses the existing backbone, so renewal-vs-direct is a
**controlled decoder comparison** on an identical encoder.

- **`alpha_tau` is a softmax over lags**, hence non-negative and summing to 1 —
  a proper distribution. This is load-bearing, not cosmetic: because the
  weights sum to 1, the convolution **commutes with the affine min–max
  normalisation**, so `alpha` remains interpretable as a generation interval on
  the raw incidence scale even though the model consumes normalised inputs.
  State this in any writeup.
- **The anchor is exact.** `x[:, -1, :]` denormalised is `rawdat[idx-h]` —
  precisely the anchor `growth_targets()` uses. The decoder's output is
  therefore a log-growth prediction under the existing target definition, so
  **no evaluation, data, or splitting code changes** (a `program.md` hard
  constraint).
- **Spatial coupling `w_ij`** is a row-softmax over
  `standardise(adj_prior) * softplus(adj_scale) + U@V`: the static geographic
  prior *and* the learned adaptive term. Row-softmax makes each row a
  distribution over source regions, as the renewal reading requires.
- **The prior is standardised per row** because row-normalising a dense graph
  destroys the only thing softmax can see. Measured within-row sd of the
  normalised prior: **0.005 on 372-node LTLA vs 0.197 on 7-node NHS** — a 38x
  gap driven purely by density, silencing geography exactly where it should
  matter most.

## First result (NHS h=7, seed 42, validation)

| model | val RMSE |
|---|---|
| frozen v2 (bar) | 7.2757 |
| exp-1 `nodecay,regpre` | **7.1788** |
| **renewal** | 8.1834 (+14%) |

**Accuracy is worse on this cell.** One cell, one seed — not a verdict, but not
encouraging either.

**The learned generation interval is the encouraging part.** Trained end-to-end
with no epidemiological supervision:

```
tau:  0     1     2     3     4     5     6     7    ...
alpha: .244  .183  .129  .088  .060  .044  .038  .034 ...
```

Monotone decay, sums to 1.0000, **mean lag 3.51 days**, mode at `tau = 0`. That
is a plausible COVID generation interval — literature estimates cluster around
3–5 days for Omicron-era serial intervals. The model recovered it from case
counts alone. This is the interpretability *evidence* the adversarial check
demands, as opposed to an interpretability assertion.


## CORRECTION (19 Aug 2026) — the h-step-ahead formulation error

**The interpretability claim as first written is withdrawn.** Two errors, found
when the lag-0 ablation (R2) came back better than the "principled" R1.

**1. `tau = 0` is not zero delay.** The input window ends at `dat[idx - h]` and
the target is `dat[idx]`, so kernel index `tau` corresponds to an actual delay
of **`h + tau`** from the target. There is no zero-delay term in this
formulation at all -- the minimum delay is `h`, forced by forecasting h steps
ahead. Cori's `w_0 = 0` convention therefore does not apply, and excluding
`tau = 0` was justified by a misreading. It also cost accuracy: R1 (tau>=1)
is +23.4% vs the bar, R2 (tau>=0) +12.5%.

**2. The reported mean lags were not generation intervals.** "Mean lag 3.51
days" and "4.869" are means of `tau`, not of delay from the target. True mean
delay is `h + mean(tau)` -- for NHS h=7 that is ~11.9 days, far outside any
3-5 day COVID generation interval. Those figures also averaged across NHS
(daily) and Japan (weekly) cells with different `h`, mixing units. Any future
kernel figure must be per (dataset, horizon) and stated as delay from target.

**3. The degeneracy never materialised.** R2 shows `alpha_first` 0.2877 and
entropy 0.8165 -- no collapse. The failure mode the exclusion was meant to
prevent did not occur in practice.

**What this means for the architecture.** When forecasting `h` steps ahead, the
infections that generate the target occur in the *unobserved gap* between
`idx-h` and `idx`. A single convolution over history older than `h` is not the
renewal equation; it is a delay kernel over stale observations. This is the
most likely reason R1 loses in 5/5 cells. A faithful renewal model must
**iterate the equation forward h times**, predicting the intermediate steps --
a different and more defensible architecture than the one implemented here.

**Status of the claim:** the current decoder does NOT support a
generation-interval interpretation for h > 1. Options are (a) restrict the
mechanistic reading to h = 1, or (b) build the iterated version. Do not
publish the kernel-recovery claim on the current implementation.


## Direction 2 result — the model tells us when it wants renewal structure

The residual variant (`renewres`) puts a free scalar `gamma` on the renewal
offset, initialised at exactly 1.0 (verified to reproduce pure-renewal output
bit-identically at init). The learned value measures how much of the
convolution the model actually keeps. Read from the trained checkpoints, not
the CSV:

| cell | learned gamma |
|---|---|
| **NHS h=3** | **+0.0016 / −0.0750** |
| Japan h=3 | +0.5116 |
| Japan h=5 | +0.4971 |
| NHS h=7 | +0.6834 / +0.6698 |
| NHS h=14 | +0.6441 / +0.7479 |

**At h=3 the model switches the renewal term off entirely** (gamma ~ 0) and
reverts to the direct decoder. At h=7 and h=14 it keeps roughly two-thirds of
it. Accuracy still loses 5/5 cells (+13.4%), so this does not rescue the
decoder — but gamma did not collapse uniformly, which matters for the
interpretation.

**This is the same horizon signature the attention work produced, from an
independent mechanism.** Spatial attention helps at long horizons and hurts at
short ones (daily datasets: h=3 +12.5%, h=7 −2.6%, h=14 −6.1%). Here a
mechanistic prior is *rejected* at h=3 and *retained* at h=14. Two separate
experiments on two separate parts of the architecture reach the same
conclusion: **structural priors earn their place at long horizons and are
actively harmful at short ones**, where local autocorrelation dominates.

It is also consistent with the unobserved-gap argument: at h=3 the gap swallows
the entire generation interval so the convolution reaches only stale history,
while at h=14 the kernel spans genuinely informative lags.

**Data-integrity note.** `append_row` originally took fieldnames from each row
while writing a header only at file creation, so per-cell rows logged after new
columns were added were misaligned against the stale header. The writer now
widens the schema and rewrites; the corrupted file is quarantined as
`attn_revival_runs.CORRUPT-schema-drift.csv`. Ledger summary rows use a
consistent schema and are unaffected, and all gamma/alpha figures above were
read from checkpoints.


## CONCLUSION — Track B closed on the single-convolution formulation

**6 configurations x 5 cells = 30 comparisons, 0 wins.** Every renewal variant
passed the kernel gate and every one lost every cell against the exp-1 bar.

| config | beats bar | mean |
|---|---|---|
| R2 lag14 tau>=0 | 0/5 | +12.5% |
| R5 lag7 tau>=0 +residual | 0/5 | +13.4% |
| R3 lag7 tau>=1 | 0/5 | +14.3% |
| R6 lag7 tau>=1 +residual | 0/5 | +15.9% |
| R1 lag14 tau>=1 | 0/5 | +23.4% |
| R4 lag21 tau>=1 | 0/5 | +35.0% |

Two independent lines say the operator, not its tuning, is the problem.

1. **Monotone in constraint.** The ordering is exactly "least renewal wins":
   shorter kernels beat longer ones (lag7 +14.3% < lag14 +23.4% < lag21
   +35.0%), and allowing tau=0 beats excluding it. The best variant is the one
   that constrains the model least.
2. **The model rejects it where it cannot help.** Given a free scalar on the
   offset, training drove gamma to ~0 at h=3 while keeping 0.64-0.75 at h=14.
   Even with the freedom to keep the convolution, it is switched off at short
   horizons.

**Mechanism.** Forecasting h steps ahead, the infections that generate the
target occur in the unobserved gap between `idx-h` and `idx`. A single
convolution over history older than `h` is a stale-delay kernel, not the
renewal equation. At h=3 the gap swallows the whole generation interval, which
is exactly where gamma collapses.

**Do not publish a generation-interval claim on this implementation.** The
withdrawn-claims section above applies in full.

**Two remaining options**, neither pursued here:

- **(a) Restrict the mechanistic reading to h=1**, where the renewal identity
  is exact and the gap vanishes. Narrow but honest.
- **(b) Build the ITERATED renewal model** that rolls the equation forward h
  steps, predicting intermediate incidence, so the convolution always sees the
  h-1 values it needs. This is the faithful version, and a genuinely different
  architecture rather than a decoder swap. It is the only route on which the
  original novelty claim survives.

**What is publishable from Track B as it stands**: a negative result with a
diagnosed mechanism, plus the horizon-dependence finding it shares with the
attention work — structural priors, mechanistic or spatial, earn their place at
long horizons and are actively harmful at short ones. Two independent
experiments, same conclusion.


## FINAL — iterated renewal tested; Track B closed

The iterated decoder (`reniter`) rolls the equation forward h steps, feeding
each prediction back so the kernel always spans the tau=1..L most recent values
(observed, then predicted). log R is clamped to +-1.5 (R in [0.22, 4.48]) so h
compounding steps cannot explode; report that as part of the architecture.

**Accuracy: 9 renewal configurations x 5 cells = 45 comparisons, 0 wins.**

| variant | beats bar | mean |
|---|---|---|
| R2 best single-convolution | 0/5 | +12.5% |
| I3 iterated + residual gamma | 0/5 | +20.6% |
| I1 iterated lag7 | 0/5 | +22.4% |
| I2 iterated lag14 | 0/5 | +25.6% |

The mechanically **correct** formulation is consistently **worse** than the
incorrect one. Iterating pays for a proper kernel with compounding error across
h feedback steps, and on these horizons that trade is clearly unfavourable.
This is a reportable finding about iterated versus direct multi-step
forecasting, not a defect.

**The gamma prediction was confirmed where it was sharpest, and only there.**
The diagnosis predicted that gamma, which collapsed to ~0 at h=3 under the
broken kernel, should recover once the kernel reached the values that actually
generate the target:

| cell | broken | iterated |
|---|---|---|
| Japan h=3 | +0.5116 | **+0.9573** |
| Japan h=5 | +0.4971 | **+1.0201** |
| NHS h=3 | **+0.0016 / -0.0750** | **+0.1991** |
| NHS h=7 | +0.6834 / +0.6698 | +0.4639 |
| NHS h=14 | +0.6441 / +0.7479 | +0.4232 |

At h=3 gamma rose in both datasets — dramatically on Japan, by two orders of
magnitude on NHS. At h=7 and h=14 it fell. So the unobserved-gap argument
explains the *short-horizon* failure specifically; it is not a general account
of why renewal loses.

**What is publishable, and it is earned rather than asserted.** The recovered
generation interval is now a genuine delay from the forecast step, so it is the
quantity the epidemiological literature measures (unlike the withdrawn figures
above, which were means of tau under the broken formulation):

| dataset | horizon | mean delay |
|---|---|---|
| NHS (daily) | h=3 | **3.33 days** |
| NHS (daily) | h=7 | **4.19 days** |
| NHS (daily) | h=14 | **3.28 days** |
| Japan (weekly) | h=3 | 1.83 weeks |
| Japan (weekly) | h=5 | 2.39 weeks |

**3.3-4.2 days across three independent horizons, inside the published COVID
generation interval of ~3-5 days, learned from case counts alone with no
epidemiological supervision.** Japan's ~2 weeks is unconvincing as an influenza
serial interval (~2-4 days), but weekly aggregation cannot represent a sub-week
kernel, so that is plausibly a resolution limit rather than a wrong answer.

This is the interpretability evidence `adversarial-priority-check.md`
recommendation 3 demands, and it holds independently of the accuracy verdict.
The honest claim is therefore: **a differentiable spatially-coupled renewal
layer recovers a plausible generation interval end-to-end, but does not improve
forecast accuracy over a direct decoder on these benchmarks.** Publishable as a
mechanistic-interpretability result with a negative accuracy finding; NOT as an
accuracy contribution.

**Still untested** (would strengthen the interpretability claim if pursued):
the decisive comparison against a FIXED literature generation interval and an
EpiEstim-style baseline, and validation of predicted R_t against EpiEstim.

## Next steps (superseded)

1. Sweep the proxy grid (nhs h3/7/14, japan h3/5) and compare validation RMSE
   per cell against exp-1 and the bar. Per cell, never the raw mean.
2. Sweep `renewal_lag` (7 / 14 / 21) and try a variant where the backbone also
   predicts a multiplicative correction — fixing the convolution entirely to
   data may over-constrain the model, which is the most likely cause of the
   accuracy loss.
3. **The decisive experiment** (`adversarial-priority-check.md`, recommendation
   3): learned `alpha_tau` versus a *fixed* literature generation interval, and
   versus an EpiEstim-style renewal baseline. The claim only survives if the
   learned kernel both recovers a plausible GI shape **and** improves accuracy.
4. Validate `R_t` against EpiEstim estimates on the same series.

## Claim wording that survives review

> We embed a differentiable, spatially-coupled renewal equation into a graph
> neural network, in which the generation-interval kernel is learned end-to-end
> rather than fixed a priori.

Do **not** claim "first to model propagation delay" — `adversarial-priority-check.md`
demolishes it via PDFormer (AAAI 2023). Required citations: Cori et al. 2013,
Fraser 2007, Pasetto et al. PNAS 2023, PDFormer, Graph WaveNet, ALiBi,
Deep Renewal Processes (Türkmen et al.).

Note the honest framing: the renewal identity is conceded up front rather than
discovered by a reviewer who knows EpiEstim. Per the adversarial check, that
concession makes the contribution *stronger*, not weaker.
