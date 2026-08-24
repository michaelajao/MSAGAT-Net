# MSAGAT-Net — Research Ledger

Status as of 19 August 2026. Covers what is established, what is decided-but-unacted, and what has not been touched at all.

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
| E6 | Quantile heads well calibrated on weekly ILI, badly under-cover on UK data (LTLA 90% interval → 72%). Conformal fixes it: LTLA h=3 cov90 **0.722 → 0.916** with per-region conformal, and WIS falls too. [The previously quoted "coverage error 0.433 → 0.026" does not reproduce from `conformal_metrics.csv` under any tested definition — restated in cov90 terms 24 Aug 2026.] Per-region conformal beats attention-weighted — which follows directly from E3. | Strong. | Yes — "first calibrated probabilistic forecaster on these datasets". |
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
- **Co-author communication.** The headline result moved from 23.5% to 5.2%. Palade, He, Wark, Mousavi and Mukandavire signed off on the AIIM version. They need to hear this from you, early, framed as an audit you ran and a correction you found — not discovered later in a revised draft.
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
