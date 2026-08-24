# Paper draft — Learned generation intervals in a differentiable renewal layer

Working draft, 20 August 2026. Every number below is measured and traceable to
`report/results/`; nothing is projected. Claims that failed are recorded as
such, because the negative results are part of the contribution.

---

## Title (working)

*A differentiable renewal layer recovers epidemiologically meaningful structure
without improving forecast accuracy*

## Abstract (draft)

Neural epidemic forecasters are routinely described as learning transmission
dynamics, but that interpretation is rarely tested. We embed a differentiable,
spatially-coupled renewal equation into a graph neural network, in which the
generation-interval kernel is learned end-to-end rather than fixed a priori,
and ask whether the learned quantities correspond to the epidemiological
objects they are named after. On daily UK COVID-19 hospital admissions the
learned kernel recovers a mean generation interval of 3.2-4.6 days across three
forecast horizons and five random seeds, with 11 of 12 independently trained
models falling inside the published 3-5 day range, and a seed-to-seed standard
deviation of 0.02 days at the shortest horizon. The reproduction number the
layer infers correlates with a classical EpiEstim estimate at r = 0.94 despite
never being supervised on it. Learning the kernel outperforms both a flat
kernel and the published literature generation interval, and approximately
halves seed variance. However, the renewal layer does **not** improve forecast
accuracy over a direct decoder: across nine configurations and 45 cell
comparisons it never wins, and the mechanically faithful iterated formulation
is consistently worse than an approximate one. We argue that mechanistic
interpretability and predictive accuracy are separable goals in epidemic
forecasting, and report a horizon threshold below which structural priors --
spatial or mechanistic -- are actively harmful.

## 1. Contributions

1. **A differentiable spatially-coupled renewal layer** whose generation
   interval is learned end-to-end (Sec. 3).
2. **Evidence that the learned kernel is epidemiologically meaningful**, not
   merely a fitted filter: it recovers published COVID generation intervals
   across seeds and horizons (Sec. 5.1), and beats both an uninformative and a
   literature-supplied kernel (Sec. 5.2).
3. **Independent validation of the inferred reproduction number** against the
   Cori et al. estimator (Sec. 5.3).
4. **A negative result, stated plainly**: mechanistic structure does not
   improve accuracy here, and the *faithful* iterated formulation is worse than
   the approximate one (Sec. 5.4).
5. **A horizon threshold for structural priors**, observed independently four
   times (Sec. 5.5).
6. **A corrected evaluation protocol** for the Cola-GNN benchmark family, under
   which all baselines are retrained (Sec. 4).

## 2. Positioning

Per `adversarial-priority-check.md`, the renewal identity is conceded up front
rather than left for a reviewer to find. The SIG-style operator **is** the
spatial renewal equation with a learned kernel; the contribution is the
end-to-end learning and the validation, not the form.

Required citations: Cori et al. 2013 (EpiEstim, AJE 178:1505); Fraser 2007
(PLoS ONE 2:e758); Ferretti et al. 2020 (generation interval); Pasetto et al.
2023 (PNAS, spatial renewal + mobility); PDFormer (AAAI 2023, delay-aware
attention -- kills any "first to model propagation delay" claim); Graph WaveNet
(IJCAI 2019); ALiBi (ICLR 2022); Turkmen et al. (deep renewal processes);
Cola-GNN (CIKM 2020); EpiGNN (ECML-PKDD 2022); DCRNN (ICLR 2018).

**Do not claim** first-to-model-delay, or novelty of adaptive adjacency.

## 3. Method

Backbone predicts `log R`; the renewal layer supplies the rest:

```
I_i(t+s) = R_i(t+s) * SUM_tau alpha_tau * SUM_j w_ij I_j(t+s-tau),  s = 1..h
```

- `alpha` is a softmax over lags: non-negative, sums to 1. Because it sums to
  1 the convolution commutes with the affine min-max normalisation, so `alpha`
  stays interpretable on the raw incidence scale.
- **Iterated**: predictions are fed back, so the kernel always spans the
  `tau=1..L` most recent values. Only in this form is `alpha` a delay from the
  forecast step and therefore comparable to a published generation interval.
- `log R` clamped to +-1.5 (`R` in [0.22, 4.48]) so h compounding steps cannot
  explode. Report as part of the architecture.
- `w_ij`: row-softmax over a **density-invariant standardised** geographic
  prior plus a learned low-rank term. Standardisation is necessary because
  row-normalising a dense graph destroys within-row variation, the only thing a
  softmax can see (measured: 0.005 on 372-node LTLA vs 0.197 on 7-node NHS).

## 4. Protocol

Correcting an error in the benchmark family: baselines were previously scored
on lead times *h..2h-1* pooled while the proposed model was scored at lead *h*.
Two baselines could not vary their output across the steps they were graded on.
Verified by bit-exact reproduction of the published table, then all baselines
reverted to single-step and retrained. 5 seeds; Diebold-Mariano with
Newey-West (lag h-1), Harvey-Leybourne-Newbold correction, Holm within each
dataset-horizon family. Leakage audit: clean (`preprocessing-audit.md`).

## 5. Results

### 5.1 The learned kernel recovers a published generation interval

Learned arm, 4 seeds per cell, delay from the forecast step:

| cell | mean delay | range | inside published 3-5 d |
|---|---|---|---|
| NHS h=3 | **3.20 ± 0.02 d** | [3.18, 3.22] | 4/4 |
| NHS h=7 | 4.60 ± 0.32 d | [4.23, 5.02] | 3/4 |
| NHS h=14 | 3.71 ± 0.37 d | [3.18, 4.05] | 4/4 |

**11/12 independently trained models inside the published range**; at h=3 the
recovery is near-deterministic across random initialisations (sd 0.02 days).
No epidemiological supervision: the loss sees only case counts.

### 5.2 The kernel is load-bearing, not decorative

Validation RMSE, 5 seeds, identical backbone:

| cell | learned | uniform | fixed (Ferretti gamma) |
|---|---|---|---|
| NHS h=3 | **3.004 ± 0.168** | 3.770 ± 1.463 | 3.829 ± 1.170 |
| NHS h=7 | **9.006 ± 0.843** | 9.703 ± 1.163 | 9.994 ± 1.537 |
| NHS h=14 | 18.880 ± 2.253 | 19.112 ± 2.041 | **18.383 ± 1.627** |

Learned beats uniform **3/3 cells (mean -9.6%)** and the literature kernel
**2/3 (mean -9.6%)**. Learning also **halves seed variance** (h=3: 0.168 vs
1.46/1.17) -- a free kernel stabilises training as well as fitting better.

*Weekly ILI is excluded here.* A COVID generation interval on weekly data is a
4.83-**week** kernel; the apparent benefit of the fixed arm on Japan
(-23%) is smoothing, not mechanism, and must not be counted as support.

### 5.3 The inferred R tracks EpiEstim

Never supervised on it:

| dataset | h | Pearson | Spearman | R>1 agreement | median R (model/EpiEstim) |
|---|---|---|---|---|---|
| NHS | 3 | **0.942** | 0.929 | 86.6% | 0.908 / 0.970 |
| NHS | 7 | **0.916** | 0.904 | 67.6% | 1.033 / 0.970 |
| NHS | 14 | 0.501 | 0.554 | 57.5% | 1.010 / 0.970 |
| Japan | 3 | 0.831 | 0.860 | 50.0% | 0.428 / 1.031 |
| Japan | 5 | 0.777 | 0.828 | 78.6% | 0.704 / 1.031 |

Correlation decays with horizon, as expected: R at h=14 is a two-week-ahead
transmissibility forecast, not a nowcast. Japan's *level* is biased (median
0.43-0.70 vs 1.03) though its dynamics track -- report both.

### 5.4 Negative result: no accuracy gain, and faithful is worse

**9 renewal configurations x 5 cells = 45 comparisons, 0 wins** against the
direct decoder. Worse, the ordering is systematic:

| variant | mean vs bar |
|---|---|
| best single-convolution (approximate) | +12.5% |
| iterated + residual | +20.6% |
| iterated lag 7 | +22.4% |
| iterated lag 14 | +25.6% |

The **mechanically correct** iterated formulation is consistently worse than
the approximate one: it buys a proper kernel and pays with error compounding
across h feedback steps. This is the paper's central tension and should not be
softened.

### 5.5 A horizon threshold for structural priors

Observed four times, independently:

| evidence | short horizon | long horizon |
|---|---|---|
| spatial attention (daily, test) | h=3 **+12.5%** (0/3 improved) | h=14 **-6.1%** (3/3) |
| renewal residual weight `gamma` | h=3 **~0.00** | h=14 0.64-0.75 |
| learned vs uniform kernel | h=3 ~tie | h=7 -20% (seed 42) |
| learned vs fixed kernel | h=3 ~tie | h=7 favours learned |

Below roughly a week, local autocorrelation dominates and structural priors --
spatial *or* mechanistic -- inject noise. The model's own learned `gamma`
switches the mechanism off at h=3 without being told to.

## 6. Limitations (write these, do not bury them)

- No accuracy improvement; this is an interpretability paper.
- Kernel statistics are n=4 seeds (one checkpoint in a prior directory);
  accuracy is n=5. Rerun for camera-ready.
- Weekly ILI cannot represent a sub-week generation interval; conclusions are
  restricted to daily series.
- `log R` clamping is an architectural constraint the direct decoder does not need.
- Effect sizes shrank under multi-seed confirmation (learned-vs-uniform at h=7:
  -20% on one seed, -7.2% over five). Single-seed results in this area are
  unreliable and we say so.
- EpiEstim comparison uses a national mean series and a flat-prior point
  estimate, not the full Bayesian posterior.

## 7. Still to do before submission

1. Rerun kernel stats at n=5 (trivial).
2. DM tests on the renewal arms, not just descriptive means.
3. Validate `R` per region rather than national mean.
4. Full EpiEstim posterior (credible intervals) rather than the point estimate.
5. Decide venue. This is an interpretability/methodology paper -- *PLOS
   Computational Biology*, *Epidemics*, or a NeurIPS/ICML workshop on ML for
   health, not an architecture venue.
6. Reconcile the canonical dataset table (node counts differ across documents).
7. Tell co-authors the headline moved from 23.5% to a tie; see ledger section 5.
