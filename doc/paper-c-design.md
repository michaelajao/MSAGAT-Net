# Paper C — design document

**Status:** design only. Not started, and not to be started until Paper B's
protocol, floors and calibration tables exist.
**Version:** v0, 25 August 2026.
**Venue target:** Neurocomputing or Knowledge-Based Systems (Elsevier hybrid,
covered by the Coventry agreement).

Working name: **SAGE-Epi** — *Structured Autoregression with a Graph
residual, Epidemic*. Rename freely.

One line: *an identifiable, likelihood-based spatiotemporal forecaster whose
structured autoregressive part is interpretable and whose graph part is
provably orthogonal to it, meta-trained across outbreaks.*

---

## 1. Why this and not something else

The reads of 24–25 August (see `doc/audit-2026-08-24/`) closed most of the
obvious directions:

- **Another SIR-embedded GNN** is occupied by seven systems — STAN, MepoGNN,
  EINNs, EARTH, HeatGNN, PISID, CSTGNN. On ILI-rate benchmarks the SIR state
  is unobservable anyway: those series are consultation rates, not infections,
  and across six recurring seasons there is no monotone susceptible depletion
  for `βS` to describe. Two independent groups (Panagopoulos p. 7; Gao)
  report that simple mechanistic models on counts-only data produce "errors in
  a different scale".
- **A neural-ODE forecaster with a free-form learned adjacency** is MTGODE,
  done on 26k-timestep data with supporting theory and margins of 1–4% RSE.
  Re-running it on 348-step ILI adds nothing.
- **Better attention over a geographic prior** is Cola-GNN §3.2, EpiGNN §3.4,
  and STTGNN (KBS 2026). Saturated.

What is **not** occupied, and needs only counts + adjacency + population:

1. **Fritz's identifiability constraint** (Sci Rep 2022) — orthogonalising the
   neural part against the structured part inside one likelihood — has never
   been carried into the epidemic-GNN benchmark line. And Fritz's own GNN is
   *time-constant* (his Table 1 gives it only static node and edge
   attributes), so an ordinary time-varying version is already a strict
   improvement on his design.
2. **Cross-dataset / cross-season transfer** with counts as the only shared
   modality. Both Panagopoulos ("Our final goal is to evaluate the model on
   the second wave of COVID-19, based on the first") and Nikparvar ("use this
   pre-trained model to predict disease dynamics in other geographic contexts,
   time periods, or even similar infectious diseases such as influenza") name
   it as future work. Neither executed it.
3. **Validating learned epidemiological parameters against something
   external.** STAN learns per-region β and γ and never reports a single
   value. No paper in the set does this.
4. **Robustness to irregular or missing observations** — asserted twice by
   PAN-cODE, tested by nobody, and impossible for fixed-window discrete models
   by construction.

The axis is *identifiability, calibration and transfer*, not *which
mechanistic prior to embed*. That is orthogonal to the entire 2024–2026 line.

---

## 2. Inputs

Counts, geography and population only.

| Input | Source |
|---|---|
| `y_{i,t}` counts per region | existing datasets; **raw unsmoothed** NHS/LTLA from `EpiHealthForecast` (see `doc/audit-2026-08-24/06-data-provenance.md` §5) |
| static adjacency `A` | existing `data/*-adj*.txt` |
| population `N_i` | **to collect**: ONS mid-year (LTLA, NHS regions), e-Stat (Japan prefectures), US Census Bureau (states, HHS regions), ABS (Australian states) |
| optional `R̂_{i,t}` | EpiEstim from counts, or Paper A's renewal-decoder R |

Nothing else. No mobility, no claims, no NPI indices — the four papers that
need those are precisely the four that cannot run on this benchmark.

---

## 3. Model

Rate `λ_{i,t+h} = N_i · exp(η_{i,t+h})`, per-horizon head so the lead-*h*
protocol is preserved, with

```
η = η_struct(Z_{i,t}) + η_gnn⊥(A, X_{t−w:t})
```

### 3.1 Structured part `η_struct`

A penalised structured predictor on the design matrix `Z`:

- lagged log-rates `log((y_{i,t−k}+1)/N_i)`, k = 1..p — linear AR or a P-spline
  over the lag axis
- seasonality: Fourier terms (weekly on daily data, annual on ILI)
- a linear trend
- optionally `log R̂_{i,t}`

with a quadratic penalty `θᵀPθ`. Coefficients are reported with Wald or
Laplace confidence intervals. This is the `hhh4` / Fritz endemic–epidemic
component — the term the literature repeatedly finds is doing the work.

### 3.2 Graph residual `η_gnn`

A time-varying spatial residual: two-hop message passing on `Â = D⁻¹(A+I)`
plus a rank-*r* learned correction `U Vᵀ`, over node features consisting of
the same window of neighbours' log-rates. Output `u ∈ R^{B×N}`.

**Two lessons from Paper B are built in from the start:** `U`,`V` are
**excluded from weight decay**, and their sparsity penalty is applied
**pre-softmax** so it carries gradient. Without both, the learned graph term
collapses to ~1e-36 (see `doc/audit-2026-08-24/02-architecture-audit.md` §5.2).

### 3.3 The orthogonalisation — the actual contribution

```
u⊥ = u − Z(ZᵀZ)⁻¹Zᵀu
```

No linear function of `Z` can be re-expressed by the GNN, so the
decomposition `η_struct` vs `η_gnn⊥` is unique. This makes the question
**"how much of this forecast is plain autoregression and how much is genuine
spatial spillover?"** answerable — reported as share of deviance explained,
per dataset and per horizon. That is the question the whole spatiotemporal-GNN
literature dodges by reporting only aggregate MAE.

### 3.4 Likelihood

Negative binomial with dispersion `χ` per dataset; zero-inflated (ZINB) where
zeros are frequent — LTLA and Australia. The Wang (2023) survey names
zero-inflation as *the* open problem for sparse epidemic graph data
(p. 2045), so this is a survey-endorsed target.

**Caveat to resolve first:** the LTLA/NHS matrices currently on disk are
7-day-trailing-mean smoothed and therefore non-integer. Either use the raw
series recovered in file 06, or substitute a continuous alternative
(Tweedie/Gamma) and say so explicitly.

### 3.5 Uncertainty

- **Aleatoric** from NB quantiles.
- **Epistemic** from a 5-seed ensemble (already the protocol) or a Laplace
  approximation on `θ_struct`.
- Report WIS and 50/90/95 coverage, plus **the Fritz check**: Spearman
  correlation between ensemble standard deviation and absolute error (he got
  ρ = 0.76, "grows approximately linearly with the error").
- Paper B's conformal wrapper remains available as a fallback, but the point
  is that intervals should be calibrated *natively*.

### 3.6 Transfer

First-order MAML across the six datasets — the population offset makes rates
comparable across scales:

1. **Leave-one-dataset-out**: meta-train on five, meta-test on the sixth.
2. **Within-series novel-season onset**: meta-train on early seasons, test at
   the onset of a held-out season — the low-data regime where Panagopoulos
   showed transfer matters most.

**Controls, both essential:** naive pooled pre-training (`TL_BASE`) and
target-only training. Panagopoulos found pooled pre-training was *worse* than
target-only in 12/12 cells, which is what makes the MAML result non-trivial.

### 3.7 Missing-observation robustness

Mask 10 / 25 / 50% of observations in the input window at test time. The
structured part can use whatever lags survive; fixed-window discrete
baselines cannot represent "no observation at *t*" at all.

---

## 4. Experiments

All under Paper B's protocol, reusing its floors and tables.

| # | Experiment |
|---|---|
| 1 | Main comparison against Paper B's final table (5 baselines, persistence / seasonal-naive / AR floors, Chronos-2, MSAGAT v2) |
| 2 | **Deviance decomposition** — structured vs orthogonalised spatial share, per dataset × horizon |
| 3 | Calibration — WIS, coverage, the ensemble-SD-vs-error check |
| 4 | Transfer — leave-one-dataset-out and novel-season onset, with `TL_BASE` and target-only controls |
| 5 | Missing-observation degradation curve |
| 6 | Ablations: no-orthogonalisation, no-GNN (= pure structured), no-structured (= pure GNN, expected to collapse as in Fritz), no-penalty, no-`R̂` |

---

## 5. Claims — only if the data support them

1. An **identified decomposition** of a forecast into autoregressive and
   spatial components, on six public datasets.
2. **Natively calibrated intervals** without post-hoc conformal correction.
3. **Transfer to data-scarce onsets** beating both target-only and pooled
   pre-training.
4. The structural (spatial) contribution **concentrates at long horizons and
   regime changes**, consistent with Paper B's horizon threshold (E11) and
   with the 3–5% / 23–37% split Fritz reports between calm periods and
   inflection points.

### Not claimed

- "First hybrid" — Fritz 2022.
- "First SIR-informed" — seven papers.
- "First transfer" — MPNN+TL, 2021.

The novelty is the **identifiability constraint applied to a spatiotemporal
forecaster on public epidemic benchmarks**, plus the transfer evaluation
nobody executed.

### Honest expectation

From this literature's own evidence, the achievable margin over a well-tuned
persistence baseline at short horizons on counts-only data is **3–10%**. The
defensible gains are at long horizons, at regime changes, in calibration, and
in transfer — not in average-case point accuracy. HeatGNN, a well-executed
mechanistic hybrid, gains 1.6–4.1% on the flu sets. Any headline much above
that will read as a protocol artefact.

---

## 6. Open decisions

- The name.
- Splines vs linear AR for the structured part.
- Ensemble vs Laplace for epistemic uncertainty.
- Whether `R̂` comes from EpiEstim or from Paper A's renewal decoder.
- Whether to add rolling-origin evaluation as a secondary protocol (cost).
- Whether the count likelihood uses recovered raw data or a continuous
  substitute (§3.4).

---

## 7. Autoresearch for Paper C

`program.md` is closed and merges into the research ledger; it is not
reopened. Its hard-constraint section becomes `docs/experiment-contract.md`:

- never modify evaluation, DM, or data code
- ratchet on validation only; touch test once, at the end
- one change per experiment
- verify a new token changes behaviour at construction before trusting its runs
- count and report the number of configurations tried
- every run writes a manifest

Paper C's *tuning* phase gets its own `experiments/paper-c-program.md`, on top
of the manifest / `runs_index` harness, with gate metrics fixed in advance:

- spatial-term deviance share
- 90% coverage error
- transfer gain vs `TL_BASE`
- RMSE

A configuration is kept only if it does not regress any gate. **The design
itself — the identifiability constraint, the transfer protocol, `R̂`
conditioning — is not searched; only its hyper-choices are.** The attention
campaign's lesson was that a search finds what its space contains: given only
training recipes, it returned a training recipe.

---

## Changelog

- **v0, 2026-08-25** — first draft, written from the audits of 24–25 Aug.
  Nothing implemented.
