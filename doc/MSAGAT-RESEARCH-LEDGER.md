# MSAGAT-Net — Research Ledger

Status as of 19 August 2026, with the 29 September 2026 audit in §0. Covers what is established, what is decided-but-unacted, and what has not been touched at all.

---

## 0. Audit of 29 September 2026

An independent re-check one month after the last commit (`b52f872`). Numbers
below were recomputed from committed artefacts unless marked otherwise.

**Why AIIM rejected it (AIIM-D-26-01866, 10 Aug 2026).** The editor's reason
was scope: *"The mere application of well-known or already published
algorithms and techniques to medical data is not regarded as original research
... I do not see sufficient AI-related novelty."* Reviewer #1's ten points are
now answered by E3-E21 (seeds, DM tests, threshold sweep, smoothing audit,
ablation, attention), several in the reverse of the direction the manuscript
hoped; only #8 (STAN/MepoGNN) remains open.

**Reproduced exactly:** E5 19W/10L/76T of 105 (`dm_tests_v2_best.csv`); E15
11 of 21 (`floor_comparison.csv`); E18 median MDE 33.79%, min 9.44, max 89.74
(`dm_power_v2_best.csv`: 70 of the 105 comparisons have a finite MDE; the
other 35 cannot resolve any effect size); E20 LTLA `mean_agam` deltas
(`ablation_v2.csv`); E21 0 of 18 (`sensitivity_v2.csv`).

**Corrections.**
- **E1's corrected LTLA h=7 margin of 5.2% is superseded.** It was a
  single-seed diagnostic of 10 Aug. In the final 5-seed grid LTLA h=7 is a
  tie: MSAGAT v2 86.21 vs LSTNet 85.07, p = 0.83. Quote the tie; Paper B's
  abstract already does.
- **E1 is a self-correction.** Upstream amy-deng/colagnn is single-target;
  the h..2h-1 pooling was introduced in this project's fork (`9a71d36`). This
  closes §5 "Scope of the E1 claim".
- **Paper A's decisive experiment was run** (`doc/plos-renewal/tables/arms_*`,
  `epiestim.tex`); `renewal-net-design.md`'s "still untested" is stale. But
  its `arms_test.tex` gives the learned kernel the best NHS *test* RMSE at h=3
  (2.23 vs 2.79 direct) and h=7, while the headline "0 of 45 wins" is a
  validation result against the exp-1 bar. Reconcile, with DM tests, before
  submission.
- **Paper A's GI table is not traceable to a committed artefact.**
  `gi_recovery.tex` (5-seed means 3.23 / 4.52 / 3.62 d) is computed from the
  git-ignored `save_renewal/` checkpoints by `src/scripts/paper_tables.py`.
  The 3.33 / 4.19 / 3.28 d quoted in §5c-bis are the seed-42 values and lie
  inside those ranges; persist a `gi_recovery.csv`.

**Data, verified by execution.** Chronological 60/20/20, train-only
normalisation and no target leakage confirmed by running `DataBasicLoader`;
test sizes Japan 70, US-States 72, LTLA 168, NHS 179. All data and adjacency
files are sha256-identical across MSAGAT-Net, colagnn and EpiGNN. LTLA is
UK-wide: 307 England + 32 Scotland + 22 Wales + 11 NI = 372, which settles
the 307-vs-372 discrepancy (§5). Minor: Australia has 8 negative cells
(min -20) and ends mid-rise; LTLA carries ~115 one-day reporting spikes
smeared by the 7-day mean; state them in the data-limitations text.

**Still wrong or open.** Paper B is prose only in its abstract (266 words,
over the limit); `highlights.tex` is the AIIM file and states four falsified
claims. The root `README.md` still claims O(N) and a "self-attenuating" prior
in a public repo. The baseline lead-h fix exists only as uncommitted edits in
`../colagnn` and `../EpiGNN`. No record that co-authors have been told.

**Novelty after the 2026 literature.** SpatialEpiBench (Lyu, Turcan, Wilder,
arXiv 2605.06530) finds most methods lose to a last-value forecast; TERN
(Nagashima, Funayama, arXiv 2609.18407) builds a seasonal reference into a
model on the same three Cola-GNN influenza sets. Both verified; they pre-empt
"seasonal-naive beats the family" as a headline. Still unoccupied: the
power/MDE argument (E18), the training-dynamics mechanism of the attention
collapse (E7/E19/E20), and log-growth targets as a cross-architecture effect
(E2; Bosse et al. 2023 transform the scoring scale, not the training target).
Also flagged by the review, abstract-level only, read before citing:
EpiCastBench (2605.11598), Martin et al. (2608.20980), GeoID-PINN (2608.02633,
narrows Paper C's identifiability framing), Mantis (2508.12260, dominates
cross-disease transfer).

**Recommendations.**
- Paper B: answer the editor's novelty objection by adding a remedy to the
  diagnosis: an evaluation design that *can* resolve ~5% effects (rolling
  forecast origins; a panel DM over regions, since `dm_test.dm_stat` currently
  averages squared errors across nodes into one series), with its MDE
  reported before and after. Sulis makes the retraining cheap
  (`doc/sulis-hpc.md`).
- Cheap experiments on the existing harness: zero-shot time-series foundation
  models under the corrected protocol; a lag-52 seasonal input for every
  architecture (E15's mechanism as a test, now against TERN).
- Paper A: PLOS Computational Biology or Epidemics fit a mechanistic
  interpretability result with a negative accuracy finding better than JAMIA.
- Paper C: re-scope around GeoID-PINN and Mantis before starting.

**Follow-up, 29 Sep 2026: the work moved to `michaelajao/epipanel`.** The
one-paper programme continues in a new private repository that carries this
repository's evaluation code, not its model, from commit `4bc7ed7`. Its Gate 1
re-scores this repository's 850 fixed-protocol archives and reproduces
exactly the DM grid (19/10/76), every floor RMSE, and the 29 Sep probes. The
port found three things.

- **A panel DM over regions adds no power; the recommendation above is
  withdrawn.** For the mean of a panel differential with arbitrary
  cross-regional dependence, the Driscoll–Kraay test equals Newey–West on the
  regional mean, which is what `dm_stat` already computes (checked
  numerically in epipanel's tests). Power has to come instead from four
  sources:
  - longer rolling test spans;
  - seed ensembles;
  - region-scaled losses;
  - pooling effects across cells.

  The design-stage MDE, computed from validation data for two trained models,
  is 22.5% on the fixed split and 14.2% under rolling origins (medians over
  cells).
- **`prob_eval.wis_components` swaps the labels of the overprediction and
  underprediction terms.** Total WIS is unaffected, but the two decomposition
  columns of `prob_metrics.csv` and `conformal_metrics.csv` are each other's.
- **`conformal.py` feeds each test score back one step after its target.** A
  lead-h forecast should see it only h steps later. Re-run with h-step
  feedback (identity kernel, all 105 v2 archives; the one-step version
  reproduces `conformal_metrics.csv` exactly):
  - LTLA h=3 coverage is 0.914 rather than 0.916, so Paper B's "0.72 → 0.92"
    stands.
  - Long horizons were flattered. At h=14, calibrated WIS is now worse than
    raw on LTLA (49.5 vs 48.1) and on NHS (6.42 vs 6.33). Calibrated 90%
    coverage is 0.843 on NHS h=14 and 0.860 on US-States h=15.
  - Over all cells, coverage is 0.900 rather than 0.907, and WIS is 166.3
    rather than 165.6 (raw 172.4).
- **Australia-COVID is JHU active cases, not daily new cases.**
  `audit-2026-08-24/06-data-provenance.md` §3 is wrong on this point. Every
  one of the 8 × 556 cells equals confirmed − deaths − recovered, rebuilt
  from the JHU files EpiGNN ships in `data/`:
  - columns in alphabetical order (ACT, NSW, NT, QLD, SA, TAS, VIC, WA);
  - 27 Jan 2020 to 4 Aug 2021.

  JHU zeroes New South Wales recoveries from 1 Aug 2020, so that column is
  cumulative confirmed minus deaths for the last 368 days. That covers all of
  validation and test, where it dominates the natural-scale error. The
  reconstruction is `epipanel/scripts/australia_provenance.py`.

---

## 1. Established findings

Ordered by how much weight they can bear in a paper.

| # | Finding | Evidence strength | Bears weight? |
|---|---|---|---|
| E1 | Published baseline comparison was invalid — baselines scored on lead times *h*…2*h*−1 pooled, MSAGAT-Net at lead *h* only. Two baselines could not vary output across the steps they were graded on. | Strong. Published table reproduced bit-exactly, then baselines reverted to single-step and retrained. | Yes — this is a protocol correction, the most defensible contribution in the set. |
| E2 | Log-growth targets `log((y+1)/(y_anchor+1))` help across architectures — 15 of 23 baseline cells, up to −51.9% (Cola-GNN, Australia). | Strong, multi-architecture. | Yes — transferable design principle, publishable independently. |
| E3 | The spatial attention module (EAGAM) is inert. Row entropy 1.0000; min across every row/head/test sample 0.9998. Content term contributes 0.00000, U@V contributes 0.00000 (params ~1e-36). Only the static adjacency term has spread. Ablation in v2 space: removing EAGAM is worse in 5/5 cells, mean +27.1% (the module is mean pooling, not selection). NOTE 25 Aug 2026: the earlier claim that removing EAGAM *improves* RMSE 7.95% has NO traceable artefact and the one level-space multi-seed file (`aggregated_multiseed_results.csv`, Japan-only) has the opposite sign — that figure is withdrawn; see doc/audit-2026-08-24/01-docs-and-results-digest.md C1. | Strong — mechanism *and* ablation agree. | Yes, but as a negative result. Kills the original framing. |
| E4 | v2 log-growth inversion instability fixed by validation-selected level cap (3× per-node training max). LTLA h=14: 142.9 ± 42.3 → 128.8 ± 11.9. | Adequate. | Supporting detail, not a contribution. |
| E5 | Significance: DM with Newey-West + HLN + Holm, 5 seeds. v2 vs published config 19W/8L/74T (101); vs best arm **19W/10L/74T (103)** [recomputed 24 Aug 2026 from `dm_tests_v2_best.csv`; the earlier 17/10/73 predated the LTLA h=7 runs landing]; v1 vs best arm 15W/7L/79T (101). Japan and US-States entirely ties (70–72 test points — underpowered). All losses are Australia for both v2 arms; v1 also loses to cola_gnn and lstnet on NHS h=3. LTLA h=14 has only 3 of 5 baselines until the 8 outstanding runs land. | Strong methodology, honest outcome. | Yes — answers reviewer points 4 and 5 outright. |
| E6 | Quantile heads well calibrated on weekly ILI, badly under-cover on UK data (LTLA 90% interval → 72%). Conformal fixes it: LTLA h=3 cov90 **0.722 → 0.916** with per-region conformal, and WIS falls too. [The previously quoted "coverage error 0.433 → 0.026" does not reproduce from `conformal_metrics.csv` under any tested definition — restated in cov90 terms 24 Aug 2026.] Per-region conformal beats attention-weighted — which follows directly from E3, and that consequence is now **tested rather than asserted** (25 Aug 2026, `src/scripts/calibration_table.py`, `report/results/calibration_summary.csv`): if the attention matrix is uniform, using it as a calibration kernel must reduce to uniform pooling. It does, and it does so *gradedly*, tracking the entropy E3 measured. On LTLA (entropy 1.0000) and Japan (0.9996) attention-weighted and uniform conformal agree to a relative 1e-4; the largest deviation, 1.2%, is on Australia — the dataset with the lowest measured entropy, 0.9881. Across the four datasets with a measured entropy the deviation correlates with (1 − entropy) at r = +0.87, indicative only at n = 4. Pooled over all cells, per-region conformal is best on both mean WIS (165.6) and mean |cov90 − 0.90| (0.019); attention-weighted is worst on both (197.6 and 0.028). | Strong. | Yes — "first calibrated probabilistic forecaster on these datasets". |
| E7 | **The attention sparsity regulariser is mathematically inert.** `L_attn = λ‖A_h‖₁` is applied to the row-wise softmax output. Rows are non-negative and sum to 1, so the elementwise L1 norm is constant and its gradient is zero. The manuscript claims this term "directly encourages each node to attend to a sparse subset of other nodes" — it cannot. (Source: technical review, 9 April 2026, finding 1.) | Definite, provable. | Yes — this is the *mechanism* behind E3. |

**E13 (added 25 Aug 2026) — the architecture audit.** `src/models.py`,
`train.py`, `data.py`, `evaluate.py` and `utils.py` were read line by line
against three trained checkpoints. Beyond E3/E7:

- **There is no multi-scale temporal convolution in the code.** `dilation` is
  a constructor default of 1 (`models.py:118,123`) and is never passed a value
  other than 1 anywhere in `src/`. The TFEM is a single 3-tap FIR filter whose
  16 "feature channels" are BatchNorm-thresholded copies of one scalar
  sequence. "Multi-Scale" in the model name refers only to spatial hops.
- **The MSSFM locality-biased fusion weights also collapse to zero** in every
  trained checkpoint, so hop mixing is a fixed uniform average. The
  manuscript's alpha_0 > alpha_1 > alpha_2 > alpha_3 claim does not survive
  training.
- **`_init_weights` (`models.py:935-940`) overwrites every LayerNorm and
  BatchNorm gain** with `uniform(+/-1/sqrt(d))` instead of 1.0, because the
  guard tests for `'bias'` in the parameter name. This affects every
  checkpoint in `save_all/`, `save_attn/` and `save_renewal/`.
- The QKV "low-rank bottleneck" has **more** parameters than a plain linear
  (3192 vs 3168). The "O(N) linear complexity" claim in `README.md:7` is false
  — the model is O(N^2 d).
- On LTLA, `u` and `v` are **65% of all parameters**, and every one of them is
  at 1e-25 to 1e-41.
- Parameter formula, validated exactly against three checkpoints:
  **P(N, S, h) = 7962 + 64N + 1121S + 23h** (+902 quantile head, +L renewal).

Consequence: in the exact configuration the AIIM manuscript describes, the
trained model is an MLP over one 3-tap temporal filter, plus fixed uniform
spatial mean pooling, plus a fixed uniform hop average, plus a damped-trend /
AR blend. That model ties Cola-GNN and EpiGNN on 74 of 103 DM comparisons.
Full detail: `doc/audit-2026-08-24/02-architecture-audit.md`.

**E14 (added 25 Aug 2026) — two campaigns were never run.** No artefact
anywhere has `sim_mat != 'default'`, so `chunk_sensitivity` (the
100/200/250 km adjacency sweep) never executed; and the only ablation rows are
2-seed April level-space runs, so `chunk_ablation` never executed at 5 seeds
in v2 space. Reviewer points #1 and #3 therefore remain open. Also: the
GraphWaveNet claim is unsupported — `epilearn_baselines_results.csv` has one
row (japan, h=3, seed 42, March, pre-fix) and GraphWaveNet appears in zero DM
comparisons.

**E15 (added 25 Aug 2026) — naive floors, and a seasonality failure on the
flagship dataset.** Persistence, seasonal-naive and per-node AR(4) were run
under the same protocol as every other model (lead-h, same split, same test
indices; alignment asserted against an existing archive per cell).
Artefacts: 63 archives in `report/predictions/`, summary in
`report/results/floor_comparison.csv`, generated by
`src/scripts/{naive_baselines,floor_comparison}.py`.

- **MSAGAT-Net v2 beats the best naive floor in 11 of 21 cells.** The best
  of *all* trained models (any MSAGAT arm or any retrained baseline) manages
  16 of 21. SpatialEpiBench (2026) reports "every method beats the naive
  baseline less than 50% of the time" across 11 other datasets; this
  benchmark family behaves the same way.
- **On Japan-Prefectures, a seasonal-naive forecast at lag 52 has lower RMSE
  than every trained model at every horizon** — 1022 against a best trained
  model of 1244 / 1275 / 1686 / 1709 at h = 3 / 5 / 10 / 15, i.e. the models
  are 18% / 20% / 39% / 40% worse than one line of code. **The difference is
  NOT statistically significant**: DM with Holm gives p = 0.199 / 0.203 /
  0.139 / 0.122 against MSAGAT-Net v2, because Japan has only 70 test
  points. The point estimate favours the naive floor at every horizon and
  the test cannot resolve it — which is itself a finding about the
  benchmark, and the same underpowering E5 already reported for Japan. State
  it that way; do not claim the floor "beats" the models.
- **It is genuine annual seasonality, not an artefact.** RMSE by lag on
  Japan: 13 wk 3260, 26 wk 3298, 39 wk 3200, **52 wk 1022**, 65 wk 2961,
  78 wk 2994, 104 wk 1486 — a sharp minimum at one year and at two years.
  National-mean autocorrelation is r = 0.729 at lag 52 and −0.288 at lag 26.
  The same shape appears on region785 (52 wk 992, 104 wk 952, others
  1547–2135) and state360 (52 wk 256, 104 wk 349, others 375–563). The three
  daily COVID series show no such structure — their RMSE rises monotonically
  with lag, so persistence is their hardest floor.
- **No leakage.** Seasonal-naive uses the observation at t−52, which is
  available at forecast time whenever h ≤ 52; every horizon here qualifies.
- **The mechanism is structural.** Every model in this family reads a
  20-step input window. On weekly data that is 0.38 of an annual cycle, so
  none of them can see the previous season at all. This is the Wang (2023)
  survey's open problem §3.2.5 ("Most of the networks merely focus on
  proximity, yet ignore the trend and periodicity") showing up as a
  measurable loss on the lineage's flagship dataset.
- Australia h=14 is the only other cell no trained model wins on the point
  estimate (persistence 276.7 vs 303.9, +9%).
- **The floors were then put through the DM test as a SEPARATE Holm family**
  (`dm_test.py --include-floors`), deliberately not pooled with the five
  trained baselines: the two ask different questions, and pooling would have
  enlarged the correction on the primary comparison and silently moved the
  published counts. Verified: with the floors excluded, every DM statistic
  reproduces the committed grid exactly (max |Δdm| = 0, max |Δp_raw| = 0,
  identical verdicts); the only `p_holm` change is the LTLA h=14 family
  correctly growing from 3 to 4 members as cola_gnn completed. Selftest
  still passes (seed-vs-seed 14% ≈ α, degraded 100%).
  - **baselines family: 104 comparisons, 19W / 10L / 75T.**
  - **floors family: 63 comparisons, 22W / 6L / 35T.**
  - **All six significant losses to a floor are Australia** — persistence and
    AR(4) at h = 3, 7 and 14 (e.g. h=7: ours 410.1 vs persistence 166.3,
    p = 5.0e-4). Australia is also the source of all 10 significant losses to
    trained baselines. It is now the single clearest weakness in the paper
    and needs either a diagnosis or an explicit exclusion with justification.
  - On the daily COVID series MSAGAT-Net beats seasonal-naive decisively
    (p ≤ 0.03 everywhere), as expected: those series have no annual cycle,
    so a lag-364 forecast is a poor null there. The seasonal floor only
    bites on the weekly ILI sets.

This belongs in Paper B as a first-class result, not a footnote: it is the
strongest available evidence that the benchmark family's accuracy claims
were never tested against the right null.

**E16 (added 25 Aug 2026) — Australia diagnosed, and a general
under-forecasting bias found.** The ledger has carried "no diagnosis
attempted" against Australia since 19 Aug; every significant DM loss, to a
trained baseline or to a naive floor, is on that one dataset. Script:
`src/scripts/australia_diagnosis.py`; artefact:
`report/results/australia_diagnosis.csv`. Four hypotheses tested (two others
were rejected earlier: observation noise, and a train-to-test range shift).

- **REJECTED — the model is noisy.** Prediction volatility is *below* the
  series', not above: sd(pred)/sd(true) is 0.72 / 0.46 / 0.31 at h = 3 / 7 /
  14. The model over-smooths, it does not oscillate.
- **CONFIRMED — the error is a systematic under-forecast** that grows with
  horizon: bias −6.8% / −14.6% / −23.4% of the test-period level at
  h = 3 / 7 / 14.
- **CONFIRMED — and it is not confined to the wave.** RMSE by quartile of
  the Australia test window at h=7: 226 / 172 / 226 / 701 against
  persistence 23 / 11 / 43 / 329. The model is an order of magnitude worse
  than persistence even on the three *flat* quartiles (level ~700), and
  worst on the rising final quarter (level 980, ending 1282).
- **SUPPORTED, NOT PROVEN — a regime gap at model-selection time.**
  Australia's test window peaks at **1.88× the validation maximum** and ends
  41% above where it starts, the largest such gap of the six datasets;
  Japan is next at 1.45. Those two are the only datasets where any trained
  model loses to a naive floor, and every dataset with a gap below 1.1 wins
  every cell. The gap predicts the bias across cells (Pearson r = −0.514,
  p = 0.017, n = 21). **But the 21 cells come from only 6 datasets and cells
  within a dataset share one gap value, so they are not independent and that
  p-value is optimistic.** State it as a supported hypothesis. Note the
  earlier rejected version of this test compared the test window against
  *training*; validation is the right reference, because validation is what
  early stopping and the growth-space level cap are calibrated on.

**The wider finding matters more than Australia.** The under-forecast is not
Australia-specific: five of six datasets show a negative bias growing with
horizon — Japan −21% to −45%, LTLA −2.0% to −29.5%, Australia −6.8% to
−23.4%, region785 −9.8% to −17.3%, state360 −7.7% to −12.2%; only NHS does
not (+1.7% to +12.9%). Shrinking toward the mean as the horizon grows is a
general property of this model, and Australia is simply where it costs most
because persistence is unusually strong there. This belongs in Paper B's
limitations as a measured property, and it is a direct target for Paper C,
whose likelihood formulation and log-population offset address exactly this.

**E17 (added 25 Aug 2026) — E1 confirmed from the opposite direction.**
The correction used everywhere else pulls the baselines *down* to lead-h
scoring and retrains them. `src/scripts/pooled_symmetric.py` pushes
MSAGAT-Net *up* instead: its stored lead-h prediction is replicated across
leads h..2h-1 and scored against pooled targets, which is exactly how
LSTNet and CNNRNN-Res were graded (both emitted one prediction and expanded
it across every step they were scored on). Artefact:
`report/results/pooled_symmetric.csv`, 5 seeds per cell.

Margin over the best baseline *as submitted* (both figures use the
submitted, pooled baseline numbers from `doc/archive/paper_results_final.csv`):

| cell | as submitted | symmetric, v1 | symmetric, v2 |
|---|---|---|---|
| LTLA h=3 | +13.5% | −1.4% | +2.3% |
| **LTLA h=7** | **+23.6%** | **+10.4%** | **+13.0%** |
| LTLA h=14 | +17.3% | +7.0% | −5.0% |
| NHS h=3 | −9.6% | −44.8% | −4.7% |
| **NHS h=7** | **+22.1%** | **−4.1%** | **−6.6%** |
| NHS h=14 | −7.4% | −4.8% | −27.7% |

v1 (level space) is the arm the manuscript actually describes, so it is the
like-for-like comparison; v2 adds log-growth targets and quantile heads.

- **The 23.5% LTLA headline becomes +10.4%** under symmetric scoring — the
  direction survives, the magnitude does not.
- **The 22.2% NHS headline reverses sign, to −4.1%.** Under symmetric
  scoring MSAGAT-Net is *worse* than EpiGNN in that cell.
- Pooled scoring costs MSAGAT-Net 11–31% RMSE across the six cells, which is
  the size of the handicap the baselines carried and the model did not.

Two limitations to state in the paper: pooled targets need h−1 observations
past each scored index, so the last h−1 test samples are dropped (168 → 155
on LTLA h=14, 179 → 166 on NHS h=14; counts are in the CSV); and replicating
one prediction across h leads reproduces the old protocol's *handicap*, not
a well-specified multi-step task.

**Both directions agree, so the conclusion does not depend on which way the
correction is applied.** The corrected lead-h protocol remains the headline
because it matches upstream Cola-GNN and the wider literature; this is the
appendix table that closes the argument.

**E18 (added 25 Aug 2026) — the benchmark cannot resolve the effect sizes
this literature reports.** The ledger has asked since 19 Aug for a power
calculation "so ties read as 'correctly underpowered' rather than 'no
difference found'". `src/scripts/dm_power.py` computes, for every comparison
in the grid, the smallest RMSE reduction the DM test could detect at 80%
power, using the same Newey-West variance and HLN correction as
`dm_test.dm_stat` (imported, not re-implemented). Artefact:
`report/results/dm_power_v2_best.csv`.

**Across 70 comparisons the median minimum detectable effect is 33.8% RMSE**
(best cell 9.4%, worst 89.7%).

| claimed effect | source | resolvable in |
|---|---|---|
| 5.6% | EpiGNN 2022, its own headline | **0 of 70** |
| 4.1% | HeatGNN over Cola-GNN, flu sets | **0 of 70** |
| 23.5% | this manuscript, LTLA | 27 of 70 (39%) |
| 22.2% | this manuscript, NHS | 24 of 70 (34%) |

Worst cells: state360 h=15 (MDE 89.7%), region785 h=15 (73.5%), Australia
h=14 (57.3%), Japan h=5 (56.4%). **NHS h=14 is unresolvable at any effect
size** — the detectable MSE gap exceeds the baseline MSE itself for all five
baselines. Best-powered: region785 h=3 (14.9%), NHS h=3 (15.2%), LTLA h=3
(16.0%) — still far above any effect this literature typically reports.

This explains E5 mechanically. Japan and US-States are ties in every cell
not because the models are equivalent but because nothing smaller than a
33–90% gap could have been detected there. It also explains E15: Japan's
seasonal-naive advantage is ~31%, right at the h=3 MDE of 33.3%, which is
why p = 0.199.

Caveats that must travel with the number: the MDE is conditional on each
pair's realised loss-differential variance, so it is a diagnostic of this
data rather than a design calculation; alpha is Holm-adjusted to alpha/m for
the most conservative family member, which is pessimistic; 80% power is a
convention; and the published effect sizes above come from a 50/20/30 split
with a larger test window (Japan 104 points against our 70), which improves
the standard error by roughly a fifth — not enough to lift a 5% effect above
these thresholds, but the calculation should be repeated on their split
before asserting anything about their specific results.

**For Paper B this is a first-class contribution, not a caveat.** The
family's papers compete over 4–6% differences on data that cannot resolve
better than 34% on average. That is a stronger statement about the
benchmark than any individual model comparison, and it reframes the honest
DM outcome (mostly ties) from a weakness into the finding.

**E19 (added 25 Aug 2026) — E3, E7 and E13 confirmed across every
checkpoint.** Until now these rested on a one-off measurement of a handful
of checkpoints. `src/scripts/attention_diagnostics.py` derives them from the
state dicts directly, with no forward pass, so it covers **all 625
checkpoints** in `save_all/`, `save_attn/` and `save_renewal/` in seconds on
CPU. Artefact: `report/results/attention_diagnostics.csv`. It complements
`extract_attention.py` rather than duplicating it — that script runs the
model and persists the [N, N] matrix, but its `build_args` predates
`attn_exp` so it cannot rebuild the revival checkpoints.

No forward pass is needed for the argument. Logits are
`S = qk'/sqrt(d) + UV + softplus(adj_scale)·A_norm`, and a softmax is
selective only if the within-row spread of S is O(1). Two of the three terms
come straight from the checkpoint plus the adjacency file.

| arm | n | u absmax | UV row-sd | adj logit sd | UV share |
|---|---|---|---|---|---|
| v1 (level) | 218 | 6.3e-28 | **0** | 0.070 | **0** |
| v2 (log-growth) | 161 | 3.6e-07 | 1.8e-14 | 0.104 | 1.6e-13 |
| revived `nodecay,regpre` | 135 | **1.101** | **0.938** | 0.110 | **0.858** |

The learnable graph bias is not merely small in the frozen arms — in v1 it
underflows to **exactly zero** in float32, so the term is bit-for-bit absent
from the logits. Against the O(1) spread a selective softmax needs, the only
surviving term is the *static* adjacency at 0.07–0.10.

**The fix works, and now at scale.** `nodecay,regpre` moves the learned-term
share from ~0 to 0.858 across 135 checkpoints, not just the seed-42 proxy
grid. That does not change E9's verdict — it is a training recipe, and the
5-seed test confirmation still did not generalise — but the mechanism claim
is now evidenced 135 times over.

**The density argument is confirmed.** The adjacency logit spread is
**0.0036 on 372-node LTLA against 0.137–0.187 on 7-node NHS**, a ~38× gap
driven purely by graph density. Geography is silenced exactly where the
graph is largest — the reverse of the manuscript's "self-attenuating prior"
claim.

**MSSFM's "locality-biased" fusion is uniform.** At S=2 the v1 median
softmax weights are 0.5027 / 0.4973 (spread 0.0055); at S=4 the maximum
weight is 0.271 against the uniform 0.25 (spread 0.031). The claimed
ordering alpha_0 > alpha_1 > alpha_2 > alpha_3 does not survive training in
any arm.

**The `_init_weights` bug is visible in every trained model.** LayerNorm
gains should be 1.0. Measured medians: mean ~0.000, absolute mean 0.042
(v1) to 0.129 (v2/revived), and **50% of gains are negative**. They stay
near the buggy `uniform(+/-1/sqrt(d))` initialisation with random sign, so
every normalised branch runs at roughly a tenth of unit scale with half its
channels sign-flipped.

**PPRM's persistence branch decays with the lead**, from 0.570 at h=3 to
0.009 at h=15 in level space — measured, not assumed.

**E11 UPDATE (25 Aug 2026) — recomputed, persisted, and corrected.** The
horizon-threshold numbers had only ever lived in local variables inside
`paper_figures.py`, rendered into a PNG and quoted in prose without any
artefact holding them. `src/scripts/horizon_threshold.py` now computes them
from the prediction archives and writes
`report/results/horizon_threshold.csv`. Two errors surfaced in the process:

1. The frozen arm was picking up the `attnfix` runs, which carry no
   `attn_exp` token, duplicating seeds in the pairing (NHS showed 10 seeds
   where 5 exist). Excluded explicitly.
2. The pooled delta was being read as a mean. On the daily series the mean
   at h=3 is **+117.8%**, driven entirely by Australia h=3, whose paired
   deltas range over ±350% across seeds. The median is the honest statistic
   and is what the paper should quote.

Attention side, daily series, revived `nodecay,regpre` against the frozen v2
model, paired within seed on the test split:

| h | median delta | seeds improved |
|---|---|---|
| 3 | +0.02% | 5/15 (33%) |
| 7 | −1.21% | 9/15 (60%) |
| 14 | **−6.20%** | **13/15 (87%)** |

The monotone trend holds and the h=14 figure (−6.20%) reproduces the
previously quoted −6.1%. The previously quoted +12.5% at h=3 does not: the
median is +0.02%, i.e. restoring a selective attention is **neutral** at
short horizons rather than clearly harmful, and only the fraction of seeds
improving (33%) points the same way. Quote the improved-seed fraction, which
is robust, alongside the median.

**The effect is specific to daily data.** On the weekly ILI series the
deltas are +2.93 / +2.17 / +5.12 / +1.51% at h = 3 / 5 / 10 / 15 — no
monotone pattern and consistently mildly harmful. E11 must be stated as a
daily-series finding, which is how it was originally framed.

**E5 FINAL (25 Aug 2026) — the baseline campaign is complete and the grid
is closed.** All **525 runs** (5 baselines x 21 cells x 5 seeds) finished;
the driver exited clean, 0 failures. `runs_index.csv` covers 1376 archives
with **zero metric mismatches**. Every cell now has 5 seeds for every
baseline, including the LTLA h=14 cola_gnn and dcrnn runs that were
outstanding since 19 August.

Trained-baseline families, all four arm combinations, Holm within each
(dataset, horizon):

| MSAGAT arm | baseline arm | n | W | L | T |
|---|---|---|---|---|---|
| v1 | level (as published) | 105 | 18 | 8 | 79 |
| v1 | best of level/log-growth | 105 | 17 | 7 | 81 |
| v2 | level | 105 | 20 | 8 | 77 |
| **v2** | **best (conservative)** | **105** | **19** | **10** | **76** |

Naive-floor families, tested separately: v1 18W/8L/37T, v2 22W/6L/35T.

**Headline for Paper B: 19 wins, 10 losses, 76 ties out of 105.**

- **All 10 losses are Australia** — every baseline at h=7, three at h=14, two
  at h=3. See E16 for the diagnosis.
- Wins concentrate on LTLA (8), NHS (4), US-Regions (4) and Australia (3).
- **Japan and US-States are ties in all 20 comparisons each**, with 70 and 72
  test points. Per E18 those cells could not have detected anything smaller
  than a 33-90% RMSE gap, so they are correctly underpowered rather than
  evidence of equivalence.

The honest summary the paper must carry: **MSAGAT-Net wins a plurality of
cells, ties most of them, and loses one dataset.** "Consistently outperforms"
is not supportable and never was.

**E20 (added 25 Aug 2026) — E3 confirmed behaviourally: on the largest
graph the attention softmax can be deleted without changing the forecast.**
`chunk_ablation_v2` ran 200 cells (4 arms x 10 dataset-horizons x 5 seeds),
177 fresh, 0 failures. Script: `src/scripts/ablation_analysis.py`; artefact:
`report/results/ablation_v2.csv`.

The decisive arm is **`mean_agam`**: the full attention module with
`uniform_attn=True`, so every parameter, projection, residual and norm is
identical and **only the softmax becomes a fixed 1/N**. The test suite
asserts that flipping the flag on the full model reproduces this arm to 1e-6,
so any difference is attributable to selectivity and nothing else.

Paired within seed, against the full model:

| dataset | N | E19 entropy | h | delta % | sd | max abs | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| LTLA | 372 | 1.0000 | 3 | **+0.08** | 0.83 | 0.97 | 0.812 |
| LTLA | 372 | 1.0000 | 7 | **−0.06** | 1.71 | 2.22 | 1.000 |
| LTLA | 372 | 1.0000 | 14 | **+0.80** | 1.08 | 2.24 | 0.125 |
| Japan | 47 | 0.9996 | 3–15 | −1.4 to +4.9 | 5–12 | 7–22 | 0.06–1.00 |
| NHS | 7 | 0.9937 | 3–14 | −8.2 to +0.5 | 5–26 | 9–44 | 0.31–1.00 |

**On the 372-node graph the mean absolute change is 0.31% and no horizon is
significant.** Removing the entire attention mechanism — while keeping its
parameters, its aggregation and its residual — is not detectable in the
forecast. This is the behavioural counterpart to E19's parameter evidence
and is far stronger than the original `no_agam` ablation, which removed all
spatial mixing and so could never separate "attention does not attend" from
"spatial aggregation is useless".

The deviation again scales with graph size, as both E19 and the conformal
result predict: LTLA sd ~1%, Japan ~5–12%, NHS up to 26%. On the small
graphs the attention is slightly non-uniform *and* seed noise is far larger,
so nothing is resolvable there either.

**The wider ablation picture is bleaker than the attention finding alone.**
Pooled median delta against the full model across all 10 cells:

| arm | median | interpretation |
|---|---|---|
| `no_mtfm` | **−2.53%** | removing multi-hop spatial refinement *improves* it |
| `mean_agam` | −0.85% | deleting the softmax *improves* it |
| `no_agam` | +0.24% | removing spatial mixing entirely costs almost nothing |
| `no_pprm` | +0.82% | removing progressive refinement costs almost nothing |

**No component removal costs more than 1% in the median, and two of the four
improve the model.** Taken with E13 (no multi-scale temporal convolution
exists; the hop-fusion weights are uniform) and E19 (the graph bias
underflows to zero), the architecture's spatial pathway contributes
essentially nothing that a fixed uniform mean would not. That is the
honest ablation table Paper B must print, and it is a stronger negative
result than the manuscript's original "adaptive spatial attention is
universally essential".

A caveat to state: these cells are individually noisy (per-seed sd 5–26% on
the small graphs), and per E18 the benchmark cannot resolve small effects.
The LTLA rows carry the claim because that is where seed noise is ~1%.

**E21 (added 25 Aug 2026) — the adjacency threshold does not matter, which
completes the spatial-pathway story.** `chunk_sensitivity_v2` ran 90 cells
(2 datasets x 3 thresholds x 3 horizons x 5 seeds), 0 failures, answering
reviewer point #3. The shipped matrices are the 150 km ones, so 100, 200 and
250 km are compared against them, paired within seed. Script:
`src/scripts/sensitivity_analysis.py`; artefact:
`report/results/sensitivity_v2.csv`.

**Zero of 18 cells show a significant difference** (all Wilcoxon p >= 0.125),
despite graph density changing more than three-fold:

| dataset | 100 km | 150 km (default) | 200 km | 250 km |
|---|---|---|---|---|
| LTLA | 0.179 | 0.310 | 0.451 | 0.579 |
| NHS | 0.265 | 0.388 | 0.551 | 0.592 |

On LTLA every change is between +0.11% and +2.78% with p >= 0.31. On NHS the
deltas are larger and consistently negative (up to −13.1% at 200 km, h=14),
but with per-seed sd of 12–30% none is resolvable, and per E18 that cell
cannot detect anything smaller than a ~50% effect anyway.

**This closes the loop on the spatial pathway.** E19 showed the learned graph
bias underflows to zero, leaving the static adjacency as the only attention
logit with any spread — and that it is silenced by density on large graphs
(row-sd 0.0036 on LTLA). E20 showed the attention softmax can be deleted on
LTLA with no measurable effect. E21 now shows the graph *itself* can be
changed substantially with no measurable effect. The three findings agree:
**the model's spatial pathway is inert end to end, not merely its attention
weights.**

For the paper this converts reviewer point #3 from an unanswered objection
into a supporting result: the 150 km threshold was never justified, and it
turns out not to need justifying, because nothing downstream depends on it.
State it that way rather than as a tuning study.

**Reviewer points now answered:** #4 (single seed), #5 (no significance testing), #7 (ablation inconsistency — now explained mechanistically rather than excused), #10 (interpretability speculation — resolved by deletion).

**E7 completes E3.** Taken together the story is now closed rather than merely observed: the sparsity penalty exerts no gradient (E7), weight decay pulls `u`/`v` toward zero with nothing opposing it, and the observed end state is uniform attention at entropy 1.0000 with parameters at ~1e-36 (E3), which the v2-space ablation confirms is aggregation without selection (removal costs +27.1%, so the module pools but does not attend). Three independent lines — analytical, diagnostic, ablative — agree. This is a demonstrable failure mode, not an anomaly, and it is considerably more publishable in that form.

**E3 REFINED (19 Aug 2026) — "inert" does not mean "useless", and the ablation bar is weaker, not stronger.** Measured on the v2 proxy grid (log-growth + quantiles, *validation* RMSE, seed 42, NHS h3/7/14 + Japan h3/5): removing EAGAM is **worse in 5/5 cells, mean +27.1%** (NHS h3 +55.0%, h7 +31.0%, h14 +16.0%, Japan h3 +22.3%, h5 +11.4%). This contradicts the earlier ablation result (removal *improves* RMSE 7.95%), which was measured in **level space, on test, from the older April runs**. Both can hold: they are different target spaces and different splits.

The reconciliation is mechanical. `IdentitySpatialModule` (no_agam) is a pure per-node pass-through with LayerNorm — it removes *all* spatial mixing. EAGAM with uniform attention is not doing nothing: uniform attention times values is an **unweighted spatial mean pooling**, added residually. So the module is a global mean-pooling layer wearing an attention costume. The attention claim is still dead (entropy 1.0000, U@V at 1e-36); what survives is that the *aggregation* is useful while the *selectivity* is absent.

Two consequences. (a) `program.md`'s stated bar — "the ablated model with EAGAM removed, which is stronger" — does not hold in v2 space; the operative bar for the revival campaign is **frozen v2 per cell**, which is the harder target. (b) The campaign question sharpens to: *can EAGAM be made selective enough to beat its own uniform-mean special case?* That is exactly what the program's gate (entropy ≤ 0.98, learned-term variance share ≥ 0.10) was designed to test, so the gate stands unchanged.

**Reviewer point #2 is answered in reverse.** The manuscript argued the adjacency prior *self-attenuates* under softmax shift-invariance. E3 shows the opposite: it is the only term carrying information. The theory section is not merely too narrow, it is backwards. That whole subsection has to be rewritten or cut. The April technical review flagged the same overreach independently (finding 6).

**Additional definite errors from the April technical review, status unknown:**
- Attention-loss weighting defined three inconsistent ways across the regularisation subsection, the total-loss equation, Table 2, and Algorithm 1. Reproducibility problem.
- Adaptive hop-depth rule `S = min(S_max, max(2, ⌊N/5⌋))` gives S=2 for US-Region (N=10), contradicting the stated S=4.
- EAGAM notation conflates per-head and stacked-head tensors.
- Weekly datasets described in "days ahead" rather than steps/weeks.
- Dataset naming drift (LTLA and NHS each appear under three names).

---

## 2. RESOLVED (19 Aug 2026) — the attention decision

**Outcome: fix attempted, fix works, but it is a training recipe, not novelty.**
Campaign run per `program.md`; full results in `doc/attention-revival-summary.md`.

- **Winner: `nodecay,regpre`** — exclude `u`/`v` from weight decay, and replace
  the inert post-softmax L1 with a gradient-bearing pre-softmax L1 on `U@V`.
  Gate PASS (entropy 0.854, learned-term variance share **0.780 vs ~0.00
  frozen**); beats the frozen-v2 bar in 4/5 proxy cells, mean **−6.9%**
  (validation, seed 42). A 5-seed full-grid confirmation is running.
- **All nine elaborations lost to it** (directions 2–10 plus one combination),
  and `multgate` failed the gate outright with attention back at entropy 0.9952
  while its RMSE nearly tied the winner — the exact failure the gate exists to
  catch.
- **The stated bar was wrong.** `program.md` assumed the no_agam ablation was
  stronger; in v2 space it is worse in 5/5 cells (**+27.1%**). See the E3
  refinement above. The campaign was therefore run against frozen v2, the
  harder target.
- **This does not answer the novelty objection.** The architecture is
  byte-identical; `nodecay` is standard practice and `regpre` is a bug fix. The
  contribution is the **diagnosis and documented failure mode**, not the patch.

**Consequence:** EICM is viable again in principle — conditioning SIR
parameters on attention features now means something, since those features
finally carry spatial information — but it is not the priority. The novelty
push is Track B below.

## 2b. Open decision — superseded by Track B

**The attention module: fix, freeze, or drop.**

- **Fix** — exclude u/v from weight decay, add learnable temperature. Cheap to test. If it works, the architectural story survives and EICM becomes viable again.
- **Freeze** — leave inert, write the paper around E1/E2/E6. Safe, but you are shipping a module you know does nothing, which a reviewer may well rediscover.
- **Drop** — remove EAGAM entirely. Supported by both mechanism and ablation (removal *improves* RMSE 7.95%). Cleanest, and currently the most defensible.

Freeze is the weakest option and should probably be discarded. The real choice is fix-then-decide versus drop-now. Recommendation: run one bounded autoresearch campaign (see `program.md`), timeboxed, then drop if it fails. That converts an open question into either a recovered contribution or a documented negative result.

---

## 3. Not worked on — reviewer points still open

| Reviewer point | Status | Why it now matters more |
|---|---|---|
| **#1 Novelty positioning** vs adaptive graph learning, dynamic graph transformers, bias-based attention | Untouched | Needs rewriting from scratch for the new framing anyway. The old positioning defended a module you may be about to delete. |
| **#3 150 km Haversine threshold** — unjustified, untuned, no sensitivity analysis | Untouched | Now the single most important modelling decision in the model, not a detail. If adjacency is the only informative term (E3), the threshold *is* the spatial model. A sensitivity sweep is no longer optional. |
| **#8 STAN / MepoGNN excluded** as physics-informed baselines | **Partly addressed** — GraphWaveNet added via EpiLearn (strongest addition, since it learns adaptive adjacency from node embeddings and is directly comparable to B=UV). STAN and MepoGNN excluded with a written justification (require compartmental state and OD mobility data). DASTGN and GraphLSTM crash in EpiLearn; HierST unimplemented; MSGNN has no public code. | The exclusion argument is defensible and documented. Under the new framing, however, more architectures strengthen E2 rather than threatening it, so revisit whether STAN is worth the 3–5 days of adaptation. |
| **#9 Preprocessing / smoothing** [DONE 19 Aug, clean — doc/preprocessing-audit.md] — is the 7-day rolling mean applied before splitting? Does it leak across split boundaries? Do baselines receive identical smoothed inputs? Is the split chronological, and are sliding windows prevented from crossing boundaries? | **Not audited. Flagged twice.** AIIM reviewer #9, and independently the April technical review (finding 4, and editorial finding 5 on whether the 7-day smoother is applied to weekly series). | Highest-risk open item by a distance. Two independent reviews raised it and neither has been actioned. You found one evaluation bug (E1) by looking; the same discipline has not been applied here. If smoothing precedes splitting, or windows cross boundaries, every number is affected — including the corrected ones. **Blocking.** |

---

## 4. Not worked on — identified but parked

- **EICM / MRTM modules.** Built March 2026, never integrated into a submitted version. EICM premise now questioned by E3 (see opening note). MRTM is unaffected by E3 and remains untested — multi-resolution dilated temporal convolutions have nothing to do with the broken spatial path.
- **Australia underperformance.** All DM losses are Australia, both arms. No diagnosis attempted. Either explain it or exclude the dataset with justification — an unexplained systematic loss invites a reviewer to build their rebuttal around it.
- **Why log-growth fails where it fails.** You note LTLA and NHS h=3 failures are systematic, not random. A design principle without a mechanism for its failure mode reads as a heuristic. This is the difference between E2 being a contribution and E2 being a trick.
- **Underpowered datasets.** Japan and US-States are entirely ties at 70–72 test points. Consider a power calculation stating the minimum detectable effect, so ties read as "correctly underpowered" rather than "no difference found".

### The parked architecture line

A second body of work exists and appears dormant: **EpiSIG-Net v1 (40K), v3 (18K), v5**, **ASTSI-Net / EpiSILA**, **EpiMoNet**, and the **EpiDelay-Net** design document. These were designed, implemented, and in one case evaluated across 10 seeds on Japan, Australia and Spain.

The **Serial Interval Graph** is the strongest novelty claim anywhere in this corpus:

- Epidemiologically grounded — learnable delay weights α_τ interpretable as the generation interval distribution.
- Cleanly differentiated. Cola-GNN attends at the same timestep; EpiGNN's transmission risk has no temporal offset; DCRNN diffuses over hops, not time.
- Supported by your own data: adjacent-region correlation decays 0.960 (lag 0) → 0.877 (lag 1) → 0.560 (lag 3) → 0.050 (lag 7).
- **Independent of the broken attention path.** SIG operates on lagged signals; it does not depend on EAGAM carrying signal. E3 does not touch it.

Two caveats before acting on any of this:

1. **Every EpiSIG number predates the protocol fix.** All those tables were produced under h-pooled baseline scoring. The 10-seed comparison, the Spain wins, the "vs paper baselines" claims — none are currently trustworthy. Re-run under the corrected protocol before drawing conclusions.
2. **Run the E3 diagnostic on EpiSIG v5.** It uses low-rank attention in the same codebase under the same optimiser configuration. If weight decay collapsed `u`/`v` in MSAGAT-Net, the prior that it did the same in v5 is high. Check before building on it.

Also unreconciled: EpiMoNet's momentum and phase-transition components, and EpiDelay-Net's Rt-conditioned prediction and wave phase encoder, exist only as design documents.

---

## 5. Not worked on — nobody has raised these yet

- **Scope of the E1 claim.** Critical and unresolved: are you correcting *your own* use of these baselines, or the *published protocol of the baseline family*? If the h-pooling error exists in the original Cola-GNN / EpiGNN papers, the contribution is far larger — and far more delicate. It requires checking their released code, stating the claim narrowly and factually, and probably contacting authors before submission. If it is only your harness, say so plainly and the paper is smaller but safe.
- **Co-author communication.** The headline result moved from 23.5% to a statistical tie (see §0). Palade, He, Wark, Mousavi and Mukandavire signed off on the AIIM version. They need to hear this from you, early, framed as an audit you ran and a correction you found — not discovered later in a revised draft.
- **Multiple-comparisons hygiene under autoresearch.** Selecting a config from 100 validation-scored runs is a multiple-comparisons machine. A reviewer who already objected to single-seed evaluation will be unforgiving. Protocol: ratchet on validation only, touch test once at the end, re-run survivors across 5 seeds with DM. Log the number of configurations tried and report it.
- **Venue.** The new framing — protocol correction, target-space design, calibration — is an evaluation-and-methodology paper, not an architecture paper. That is a different readership. Worth a fresh venue scan rather than resubmitting into the same family that rejected the architecture story twice.
- **Data provenance does not reconcile across documents.** Spain-COVID appears as both 17 regions and 52 regions / 122 timesteps. LTLA appears as both 307 and 372 nodes. One document describes 7 datasets, the AIIM submission uses 6, and Spain does not appear in the submission at all. Some documents report 10-seed results while the submission reports single-seed. Until a single canonical dataset table exists, cross-document claims cannot be checked.
- **Baseline citations are wrong in places.** EpiGNN appears as both ECML-PKDD 2022 and IJCAI 2023; Cola-GNN as both CIKM 2020 and KDD 2020. The April review also flagged a corrupted CausalGNN bibliography entry with duplicated authors.

---

## 5b. Track B — the novelty push (chosen 19 Aug 2026)

**Renewal-equation decoder.** The user chose this over reviving EpiSIG first,
and chose "both, density-invariant" for the adjacency prior. Design, first
result and claim wording: `doc/renewal-net-design.md`.

The insight is that two things already established here are the same object:
log-growth targets (the largest measured effect in the project, 15/23 baseline
cells, up to −51.9%) *are* the renewal equation's natural parameterisation,
since `g = log((y_t+1)/(y_anchor+1))` is `log R_t` up to the convolution term.
So the network predicts `R_t` and a learned generation-interval kernel supplies
the rest. This converts the project's best heuristic into a mechanism, and
answers the ledger's own open question about why log-growth fails where it does.

Status: implemented and training (`--renewal`). First run (NHS h7, seed 42,
validation) is **worse on accuracy** — 8.18 vs 7.18 for exp-1 — but the learned
kernel is epidemiologically plausible without supervision: monotone decay,
sums to 1.0000, **mean lag 3.51 days**, mode at lag 0. The decisive experiment
(learned kernel vs fixed literature GI vs EpiEstim) is still outstanding, and
the claim only survives if the kernel both recovers a plausible shape **and**
improves accuracy.

## 5c. Track B outcome (19 Aug 2026) — negative, with a mechanism

**Renewal decoder: 6 configurations x 5 cells = 30 comparisons, 0 wins.**
Closed on the single-convolution formulation. Full detail in
`doc/renewal-net-design.md`.

The formulation is wrong for h-step-ahead forecasting: the infections
generating the target occur in the unobserved gap between `idx-h` and `idx`, so
a convolution over older history is a stale-delay kernel rather than the
renewal equation. Two independent signals confirm the operator (not its tuning)
is at fault — the results are monotone in "least renewal wins", and a free
scalar on the offset is driven to ~0 at h=3 while retaining 0.64-0.75 at h=14.

Earlier generation-interval claims are **withdrawn**: kernel index tau is a
delay of h+tau, so the reported "mean lag ~3.5 days" figures were means of tau,
not delay, and mixed daily with weekly cells.

Remaining options: restrict the mechanistic claim to h=1, or build the iterated
renewal model that rolls the equation forward h steps. Neither pursued.

**5c-bis. Iterated renewal tested and closed (20 Aug 2026).** Rolling the
equation forward h steps makes alpha a genuine delay kernel. Accuracy: 9
configs x 5 cells = **45 comparisons, 0 wins**; the correct formulation is
consistently worse than the incorrect one (+20.6% vs +12.5%), the price of
compounding error over h feedback steps. The gamma prediction held only where
it was sharpest: at h=3 gamma rose from ~0.00 to 0.199 (NHS) and 0.51 to 0.96
(Japan), but fell at h=7/h=14.

**EARNED RESULT:** the recovered generation interval is now the quantity the
literature measures — NHS **3.33 / 4.19 / 3.28 days** at h=3/7/14, inside the
published COVID range of ~3-5 days, from case counts alone with no
epidemiological supervision. Honest claim: *a differentiable spatially-coupled
renewal layer recovers a plausible generation interval end-to-end but does not
improve accuracy over a direct decoder.* Publishable as mechanistic
interpretability with a negative accuracy finding.

**5d. The cross-cutting finding.** Track A and Track B independently produced
the same result: **structural priors earn their place at long horizons and are
actively harmful at short ones.** Spatial attention on daily data goes h=3
+12.5% (0/3 improved), h=7 -2.6% (3/3), h=14 -6.1% (3/3); the renewal residual
weight goes ~0.0 at h=3 to 0.64-0.75 at h=14. Two mechanisms, two experiments,
one conclusion. This is the most transferable thing either track produced and
belongs in the paper.

## 6. Paper framing

The original claim (novel spatial-attention architecture) is not supportable. E3 makes it indefensible, and the interpretability analysis has to come out with it.

What is supportable, and is arguably stronger for a methods venue:

1. **A corrected evaluation protocol** that fixes a real error in a benchmark family (E1).
2. **Target-space design as a transferable principle**, with evidence across five architectures (E2).
3. **The first calibrated probabilistic forecaster** on these datasets (E6).

Note what this changes: the paper's centre of gravity moves from architecture to evaluation. That is a genuine contribution and reviewers of evaluation papers are receptive to negative results — E3 becomes an asset in that framing rather than an embarrassment. It also means the honest DM outcome (mostly ties) stops being a problem, because the claim is no longer "we win".

---

## 7. Suggested order

1. **Audit preprocessing and splitting for leakage.** Blocking. Flagged by two independent reviews and still open; everything else rests on it.
2. Finish the 26 remaining baseline runs.
3. **Attention-revival campaign**, timeboxed (see `program.md`). Resolves the open decision in section 2. First experiment should fix the E7 regulariser and exclude `u`/`v` from weight decay together.
4. **Run the E3 diagnostic on EpiSIG v5** — cheap, and determines whether the parked architecture line is salvageable or carries the same defect.
5. Haversine threshold sensitivity sweep (reviewer #3).
6. Build one canonical dataset table; reconcile node counts, seed counts, and baseline citations.
7. Decide claim scope for E1; talk to co-authors.
8. Rewrite positioning and theory sections against whichever framing survives.

**On framing.** There are now two viable papers, not one. The evaluation paper (E1 + E2 + E6 + E3/E7 as a documented failure mode) is safe, defensible, and mostly written already. The SIG architecture paper is a stronger novelty claim but rests on results that all need re-running. They are not mutually exclusive — the evaluation paper could establish the corrected protocol that the SIG paper then uses. That sequencing would be unusually strong, and it is worth considering given your contract end date.
