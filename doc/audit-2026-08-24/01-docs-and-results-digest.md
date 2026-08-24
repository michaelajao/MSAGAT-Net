# 01 — Documents and results digest

Read 24 Aug 2026: every `.md` in the repository (canonical and archive),
plus every artefact in `report/results/` and `report/logs/`. Numbers marked
**[computed]** were recomputed from the CSVs during the audit rather than
copied from prose.

---

## 0. Which documents are canonical

`doc/README.md:1-28` defines the hierarchy:

| Artefact | Role | Status |
|---|---|---|
| `doc/MSAGAT-RESEARCH-LEDGER.md` | single source of truth, findings E1–E7 | canonical (dated 19 Aug 2026, `:3`) |
| `doc/attention-revival-summary.md` | Track A result | canonical |
| `doc/renewal-net-design.md` | Track B lab record | canonical |
| `doc/adversarial-priority-check.md` | novelty/priority audit + verified bibliography | canonical |
| `doc/preprocessing-audit.md` | leakage audit, clean | canonical |
| `doc/paper-renewal-interpretability.md` | Paper A content outline | superseded by `doc/plos-renewal/` |
| `doc/plos-renewal/` | Paper A LaTeX + tables | **most current** |
| `doc/elservier/` | Paper B | "rewrite pending" (`doc/README.md:9`) |
| `doc/EpiSIG-Net-v3-Design.md` | parked architecture line | results predate protocol fix |
| `program.md` | both autoresearch programmes | **CLOSED 20 Aug 2026** (`program.md:1-7`) |
| `doc/archive/*` | pre-protocol-fix analyses | **"do not cite"** (`doc/README.md:23-28`) |

The root `README.md` (`:126-142`) still advertises the AIIM submission as
"under review" and describes the architecture in the dead framing. It has
**not** been updated for any of E1–E7.

---

## 1. Established findings

### E1 — The published baseline comparison was invalid

Baselines were scored on lead times *h…2h−1* pooled while MSAGAT-Net was
scored at lead *h* only; two baselines could not vary output across the
steps they were graded on. `doc/MSAGAT-RESEARCH-LEDGER.md:13`.

Evidence strength: "Published table reproduced bit-exactly, then baselines
reverted to single-step and retrained." Restated for Paper A at
`doc/paper-renewal-interpretability.md:92-100`.

Derived hard constraint, `program.md:195-197`: "Do not modify any
evaluation code… reintroducing it would silently inflate every result."

Unresolved scope question, `doc/MSAGAT-RESEARCH-LEDGER.md:121`: is this
correcting *our own* harness or the *published protocol* of the
Cola-GNN/EpiGNN family? The pooling was introduced in our fork; upstream
Cola-GNN is single-target.

### E2 — Log-growth targets transfer across architectures

`log((y+1)/(y_anchor+1))` helps in **15 of 23 baseline cells, up to −51.9%**
(Cola-GNN, Australia). `doc/MSAGAT-RESEARCH-LEDGER.md:14`; repeated
`doc/renewal-net-design.md:10-13`, `program.md:38-42`.

**[computed]** The "23 cells" is verifiable: `report/predictions/*/` contains
`_lg` arms for exactly two baselines — `cola_gnn_lg` on 10 (dataset,
horizon) cells and `lstnet_lg` on 13 = 23. Five seeds each: 50 + 65 = 115
npz files.

Space/split: **test**, baselines retrained in log-growth space, 5 seeds.

Open gap, ledger `:96`: "Why log-growth fails where it fails" (LTLA and NHS
h=3 failures are systematic, not random) is unexplained. Flagged as "the
difference between E2 being a contribution and E2 being a trick."

### E3 — EAGAM (spatial attention) is inert

Row entropy **1.0000**; minimum across every row/head/test sample **0.9998**;
content term contributes 0.00000; `U@V` contributes 0.00000 with parameters
**~1e-36**; only the static adjacency term has spread.
`doc/MSAGAT-RESEARCH-LEDGER.md:15`; restated `doc/attention-revival-summary.md:143-147`.

Original ablation (level space, test, April runs): removing EAGAM
**improves** RMSE **7.95%** on average; no-op on LTLA (−0.1% to +1.4%).

**E3 REFINED (19 Aug), v2 space:** removing EAGAM is **worse in 5/5 cells,
mean +27.1%** — NHS h3 +55.0%, h7 +31.0%, h14 +16.0%, Japan h3 +22.3%,
h5 +11.4%. `doc/MSAGAT-RESEARCH-LEDGER.md:25`; `doc/attention-revival-summary.md:24-27`.

**[computed]** Verified from `report/results/attn_revival_ledger.csv` row
`baseline-noagam`: 2.8285→4.3832, 7.2757→9.5302, 19.8442→23.0250,
627.7834→767.8387, 604.6056→673.2641 ⇒ mean +27.1%. Exact.

Reconciliation (`ledger:27`): `IdentitySpatialModule` removes *all* spatial
mixing; uniform attention × values is **unweighted spatial mean pooling**
added residually. Attention is **non-selective, not useless** — the
aggregation earns its place, the selectivity does not.

Reviewer point #2 answered in reverse (`ledger:31`): the manuscript claimed
the adjacency prior *self-attenuates*; E3 shows it is the *only* term
carrying information. "The theory section is not merely too narrow, it is
backwards."

### E4 — Log-growth inversion instability fixed by a level cap

Cap = 3× per-node training max. LTLA h=14: **142.9 ± 42.3 → 128.8 ± 11.9**.
`ledger:16`. Graded "Adequate. Supporting detail, not a contribution."
Locked as a hard constraint at `program.md:203-204`.

### E5 — Significance testing (DM + Newey-West + HLN + Holm, 5 seeds)

Ledger numbers (`ledger:17`): v2 vs published config **19W/8L/74T**; v2 vs
best arm **17/10/73**; v1 vs best arm **15/7/79**. "Japan and US-States
entirely ties (70–72 test points — underpowered). All losses are Australia."

**[computed] against the CSVs as they stand today:**

| file | W | L | T | N | matches ledger? |
|---|---|---|---|---|---|
| `dm_tests_v2_level.csv` | 19 | 8 | 74 | 101 | yes |
| `dm_tests_v1_best.csv` | 15 | 7 | 79 | 101 | yes |
| `dm_tests_v2_best.csv` | **19** | **10** | **74** | **103** | **NO — ledger says 17/10/73** |

The v2-best file is dated 20 Aug 17:04, i.e. regenerated *after* the ledger
was written (19 Aug), once the LTLA h=7 cola_gnn/dcrnn runs landed. **The
ledger's E5 line is stale for the v2-best arm.**

### E6 — Calibration

Ledger claim (`:18`): quantile heads well calibrated on weekly ILI, badly
under-cover on UK data (**LTLA 90% interval → 72%**). Conformal fixes it:
coverage error 0.433 → 0.026. Per-region conformal beats attention-weighted,
"which follows directly from E3."

**[computed]** The 72% figure reproduces exactly (LTLA h=3, raw, 5-seed mean
cov90 = **0.7222**). The "0.433 → 0.026" pair does **not** reproduce under
any obvious definition; nearest is LTLA h=7 raw mean summed |cov−nominal|
over {50,90,95} = 0.4238 → 0.0659 (identity). See contradiction C4.

### E7 — The attention sparsity regulariser is mathematically inert

`L_attn = λ‖A_h‖₁` applied to the **row-wise softmax output**. Rows are
non-negative and sum to 1, so the elementwise L1 norm is constant and its
gradient is zero. `ledger:19`. Source: technical review, 9 April 2026.

Sharper version, `doc/attention-revival-summary.md:138-141`: the penalty is
exactly `lambda/N`; the gradient w.r.t. attention is identically zero, **and**
the gradient on the *learnable* λ is positive so λ decays itself away —
"**Doubly dead**."

E7 completes E3 (`ledger:23`): three independent lines — analytical,
diagnostic, ablative — agree.

### E8 — Preprocessing / leakage audit: CLEAN

`doc/preprocessing-audit.md`, 19 Aug, verdict at `:5`: **"no
future-information leakage found."**

- Q1 (`:10-21`): LTLA/NHS smoothed offline with a **trailing (causal) 7-day
  mean**, established from the denominator signature (1,2,…,7,7,7 at the
  series head). Australia is **raw**; weekly ILI unsmoothed.
- Q2 (`:23-27`): min–max normalisation computed from the **training range
  only** (`src/data.py:141-159`).
- Q3 (`:29-39`): split strictly chronological. **Input windows DO cross
  split boundaries** — standard rolling-origin of the Cola-GNN family,
  causal, identical for every model; "not leakage, but the paper should
  state it explicitly."
- Q4 (`:41-47`): baselines receive **byte-identical** inputs — md5 checksums
  match across MSAGAT-Net, colagnn, EpiGNN (verified 19 Aug).

### E9 — Track A: the attention fix `nodecay,regpre`

`doc/attention-revival-summary.md`, `ledger:42-62`.

Winner: exclude `u`/`v` from weight decay + replace the inert post-softmax
L1 with a gradient-bearing **pre-softmax** L1 on `U@V`. Gate PASS: entropy
**0.854**, learned-term variance share **0.780 vs ~0.00 frozen**. Beats the
frozen-v2 bar in **4/5 proxy cells, mean −6.9%** (validation, seed 42).

All nine elaborations lost; `multgate` failed the gate at entropy **0.9952**
while nearly tying on RMSE — "on an RMSE-only protocol it would have been
recorded as a success" (`attention-revival-summary.md:64-72`).

Configuration count for multiple-comparisons reporting: **11 configs scored,
13 training campaigns** (2 invalidated by a harness bug and re-run).

Secondary findings (`:84-106`): interior optimum in selectivity (accuracy
degrades on *both* sides of uv_share ≈ 0.78 — the gate is "a floor, not an
objective"); whole-vector rescaling fails (temp +9.1%, scorenorm +9.8%);
rank 8 already right (rank 2 +22.2%, rank 16 +17.1%); complementarity does
not compose; **the static prior is silenced by graph size** — within-row sd
of the normalised prior **0.005 on 372-node LTLA vs 0.197 on 7-node NHS, a
38× gap driven purely by density**.

**5-seed test confirmation: the fix DID NOT GENERALISE** (`program.md:3-5`,
`doc/README.md:16`). Note the summary file at `:12-13` still says the
confirmation "is running" — see contradiction C5.

### E10 — Track B: renewal decoder, 0/45 on accuracy, interpretability earned

Single-convolution formulation: **6 configs × 5 cells = 30 comparisons, 0
wins** (`doc/renewal-net-design.md:168-180`). Ordering exactly "least
renewal wins": lag7 +14.3% < lag14 +23.4% < lag21 +35.0%; allowing τ=0
(+12.5%) beats excluding it (+23.4%).

Iterated formulation: **9 configs × 5 cells = 45 comparisons, 0 wins**
(`:226-234`). The **mechanically correct** iterated form is **consistently
worse** than the incorrect one (+20.6/22.4/25.6% vs +12.5%) — the price of
compounding error over h feedback steps.

EARNED interpretability result (`:260-284`): recovered mean delay from the
forecast step — NHS **3.33 / 4.19 / 3.28 days** at h=3/7/14, inside the
published COVID generation interval of ~3–5 days, **from case counts alone
with no epidemiological supervision**.

Honest claim wording (`:280-284`): *"a differentiable spatially-coupled
renewal layer recovers a plausible generation interval end-to-end, but does
not improve forecast accuracy over a direct decoder on these benchmarks."*

Multi-seed update (`doc/plos-renewal/tables/gi_recovery.tex`, n=5): h=3
**3.23 ± 0.06 d** [3.18, 3.33] 5/5 in range; h=7 **4.52 ± 0.34 d** [4.19,
5.02] 4/5; h=14 **3.62 ± 0.37 d** [3.18, 4.05] 5/5. **14/15 total.**

### E11 — The cross-cutting horizon-threshold finding

**"Structural priors earn their place at long horizons and are actively
harmful at short ones."** `ledger:188-194`; `renewal-net-design.md:147-158`;
tabulated `paper-renewal-interpretability.md:171-182`.

| evidence | short horizon | long horizon | source |
|---|---|---|---|
| spatial attention (daily, **test**, 5-seed) | h=3 **+12.5%** (0/3 improved) | h=7 −2.6% (3/3), h=14 **−6.1%** (3/3) | ledger `:191-192` |
| renewal residual weight γ (broken kernel) | h=3 **~0.00** | h=14 **0.64–0.75** | `renewal-net-design.md:132-140` |
| learned vs uniform kernel | h=3 ~tie | h=7 −20% (seed 42) | `paper-renewal-interpretability.md:177` |
| learned vs fixed kernel | h=3 ~tie | h=7 favours learned | same, `:178` |

Ledger `:193-194`: "Two mechanisms, two experiments, one conclusion. This is
the most transferable thing either track produced."

The attention row *is* the 5-seed test confirmation of `nodecay,regpre` —
that is what "did not generalise" means: it helps only at h≥7 on daily data
and hurts at h=3.

### E12 — γ recovery under the iterated kernel

`renewal-net-design.md:241-257`:

| cell | broken | iterated |
|---|---|---|
| Japan h=3 | +0.5116 | **+0.9573** |
| Japan h=5 | +0.4971 | **+1.0201** |
| NHS h=3 | +0.0016 / −0.0750 | **+0.1991** |
| NHS h=7 | +0.6834 / +0.6698 | 0.4639 |
| NHS h=14 | +0.6441 / +0.7479 | 0.4232 |

"At h=7 and h=14 it fell. So the unobserved-gap argument explains the
*short-horizon* failure specifically; it is not a general account of why
renewal loses."

---

## 2. Dead claims

### From the AIIM manuscript / repo README

| # | Dead claim | Why dead | Where |
|---|---|---|---|
| D1 | **"up to 23.5% improvement"** | Produced under pooled *h…2h−1* baseline scoring. Corrected headline is **5.2%**, and after DM testing **a tie**. | ledger `:122`; `paper-renewal-interpretability.md:208` |
| D2 | **"Novel spatial-attention architecture" as the contribution** | "not supportable. E3 makes it indefensible" | ledger `:198` |
| D3 | **"The sparsity regulariser encourages sparse attention"** | Provably zero-gradient; λ decays itself away | E7 |
| D4 | **The adjacency prior "self-attenuates" (theory subsection)** | Backwards — it is the only informative term. "That whole subsection has to be rewritten or cut." | ledger `:31` |
| D5 | **All attention interpretability / heatmaps** | Entropy 1.0000, min 0.9998 | ledger `:15`, `:21` |
| D6 | **"Removing EAGAM improves RMSE 7.95%" as a stable fact** | Holds only in level space on test, April runs; v2 space is +27.1% worse | ledger `:25` |
| D7 | **`program.md`'s stated bar (no_agam is stronger)** | Wrong; operative bar is frozen v2 | ledger `:29`, `:56-59` |
| D8 | **The published baseline comparison table** | E1 | `doc/README.md:23-28` |
| D9 | **`S = min(S_max, max(2, ⌊N/5⌋))` gives S=4** | Gives **S=2** for US-Regions (N=10) | ledger `:35` |
| D10 | **Attention-loss weighting** | Defined **three inconsistent ways** across the regularisation subsection, the total-loss equation, Table 2, Algorithm 1 | ledger `:34` |
| D11 | "days ahead" for weekly datasets; head notation; dataset naming drift | April technical review | ledger `:36-38` |
| D12 | **"All three daily datasets are smoothed"** | Australia is **raw** | `preprocessing-audit.md:19-21` |
| D13 | **Baseline citations** | EpiGNN is **ECML-PKDD 2022** not IJCAI 2023; Cola-GNN is **CIKM 2020** not KDD 2020; CausalGNN bib entry corrupted | ledger `:126` |
| D14 | **README's "O(N) linear complexity" + "state-of-the-art accuracy"** | Contradicted by E5 (mostly ties) and by dense N×N attention. **Unactioned — README never updated.** | `README.md:7` |

### From the EpiSIG / EpiDelay-Net novelty documents (archived)

`doc/adversarial-priority-check.md` demolishes all four "first-ever" claims.

| # | Dead claim | Killer prior art | Where |
|---|---|---|---|
| D15 | "First to incorporate propagation delays into the graph structure" | **PDFormer, AAAI 2023** — "a traffic delay-aware feature transformation module"; plus Fraser 2007, Cori 2013, Pasetto PNAS 2023, TLGNN, Deep Renewal Processes. Architecturally α_τ is "a **learnable 1-D depthwise convolution over the lag axis**." | `:36-49` |
| D16 | "First to capture asynchronous regional dynamics" | Lead-lag networks (Bennett 2022); additive attention bias — Shaw 2018, T5, **ALiBi**, Graphormer | `:51-61` |
| D17 | "First to use Rt as prediction driver" | STAN 2021, CausalGNN 2022, MepoGNN 2022, EISTGNN 2025, PISID 2025 | `:63-70` |
| D18 | "First to classify epidemic phases" | Regime-switching models; mixture-of-experts; turning-point literature | `:72-79` |
| D19 | "The learned kernel recovers a GI, mean lag 3.51 days" | **WITHDRAWN by the authors.** Kernel index τ is a delay of **h+τ**; figures mixed daily with weekly cells. True mean delay for NHS h=7 was **~11.9 days**. | `renewal-net-design.md:87-122` |
| D20 | The GI reading for h>1 under single-convolution decoding | Infections generating the target occur in the **unobserved gap**; a single convolution over older history is a **stale-delay kernel**. "**Do not publish a generation-interval claim on this implementation.**" | `:113-122`, `:199-200` |
| D21 | Cori's `w_0 = 0` convention justifying τ≥1 | "excluding τ=0 was justified by a misreading. It also cost accuracy: R1 +23.4% vs R2 +12.5%." The guarded-against degeneracy never occurred. | `:96-110` |
| D22 | All EpiSIG-Net v1/v3/v5 tables, the 10-seed comparison, the Spain wins | "**Every EpiSIG number predates the protocol fix.**" | ledger `:112` |
| D23 | `archive/full_comparison_analysis.md`'s "24/24 and 15/15, 100% win rate" | Pre-fix (Feb 2026), 50/20/30 split, seed 42, h-pooled scoring | `doc/README.md:23-28` |
| D24 | "MSAGAT-Net was rejected for zero epidemiological inductive bias; EpiSIG is the fix" | The replacement's own novelty claims are D15–D18 | `archive/novelty_analysis.md:9-36` |
| D25 | "first to model propagation delay" framing, in any future paper | "is dead and must not reappear" | `program.md:257-258` |

---

## 3. The current DM grid

Source: `report/results/dm_tests_v2_best.csv` (103 comparisons, 20 Aug
17:04). Method: `src/scripts/dm_test.py`.

### How it works

- Loss differential `d_t = mean_over_nodes(e_MSAGAT,t²) − mean_over_nodes(e_base,t²)`
  on the exact same test timesteps (docstring `:1-29`).
- **DM statistic:** Newey–West Bartlett long-run variance, truncation lag
  **h−1**; Harvey–Leybourne–Newbold small-sample correction; Student-t with
  n−1 df (`:108-130`).
- **Multiplicity:** Holm–Bonferroni **within each (dataset, horizon) family**
  (`:133-142`, applied `:238-242`).
- **Seed choice:** median-RMSE seed for each model (`:20-21`, `median_seed()`
  `:103-106`); `seeds_agree` counts how many of 5 give p<α and dm<0.
- **Arm selection** (`--arms best`, default, `choose_arm()` `:81-95`): every
  baseline exists in level and `_lg` arms; `best` picks whichever has the
  **lower median-seed test RMSE**. Deliberately conservative: *"selecting
  the baseline's arm on test RMSE favours the baseline, never us."*
- **Alignment guard** (`:145-152`): pairs dropped with a WARN if `y_true`
  does not match to rtol 1e-3.
- **Selftest** (`:169-202`): seed-vs-seed significance rate must be ≤0.25 and
  degraded-prediction detection ≥0.95.

### W/L/T by dataset × horizon **[computed]**

W = significant + MSAGAT better; L = significant + baseline better;
T = not significant after Holm. α = 0.05.

| dataset | h | n_test | W | L | T | N | losses |
|---|---|---|---|---|---|---|---|
| australia-covid | 3 | 112 | **3** | **2** | 0 | 5 | cola_gnn(lg), CNNRNN_Res |
| australia-covid | 7 | 112 | 0 | **5** | 0 | 5 | **all five** |
| australia-covid | 14 | 112 | 0 | **3** | 2 | 5 | cola_gnn(lg), CNNRNN_Res, lstnet(lg) |
| japan | 3 | 70 | 0 | 0 | **5** | 5 | — |
| japan | 5 | 70 | 0 | 0 | **5** | 5 | — |
| japan | 10 | 70 | 0 | 0 | **5** | 5 | — |
| japan | 15 | 70 | 0 | 0 | **5** | 5 | — |
| ltla_timeseries | 3 | 168 | **3** | 0 | 2 | 5 | — |
| ltla_timeseries | 7 | 168 | **4** | 0 | 1 | 5 | — |
| ltla_timeseries | 14 | 168 | **1** | 0 | 2 | **3** | cola_gnn, dcrnn **missing** |
| nhs_timeseries | 3 | 179 | **4** | 0 | 1 | 5 | — |
| nhs_timeseries | 7 | 179 | 0 | 0 | **5** | 5 | — |
| nhs_timeseries | 14 | 179 | 0 | 0 | **5** | 5 | — |
| region785 | 3 | 157 | **2** | 0 | 3 | 5 | — |
| region785 | 5 | 157 | **2** | 0 | 3 | 5 | — |
| region785 | 10 | 157 | 0 | 0 | **5** | 5 | — |
| region785 | 15 | 157 | 0 | 0 | **5** | 5 | — |
| state360 | 3 | 72 | 0 | 0 | **5** | 5 | — |
| state360 | 5 | 72 | 0 | 0 | **5** | 5 | — |
| state360 | 10 | 72 | 0 | 0 | **5** | 5 | — |
| state360 | 15 | 72 | 0 | 0 | **5** | 5 | — |
| **TOTAL** | | | **19** | **10** | **74** | **103** | |

### W/L/T by baseline **[computed]**

| baseline | N | W | L | T |
|---|---|---|---|---|
| dcrnn | 20 | **6** | 1 | 13 |
| epignn | 21 | **5** | 1 | 15 |
| CNNRNN_Res | 21 | **5** | 3 | 13 |
| lstnet | 21 | 2 | 2 | 17 |
| cola_gnn | 20 | 1 | 3 | 16 |

### Which arm was selected **[computed]**

`dm_tests_v2_best.csv`: **87 level, 16 loggrowth**. The log-growth arm is
only ever chosen for **cola_gnn** and **lstnet** — the only two baselines
with `_lg` predictions on disk.

### The other two arms **[computed]**

| file | W | L | T | N | losses located |
|---|---|---|---|---|---|
| `dm_tests_v2_level.csv` | 19 | 8 | 74 | 101 | **all Australia** |
| `dm_tests_v1_best.csv` | 15 | 7 | 79 | 101 | Australia ×5 **+ NHS h=3 ×2** |
| `dm_tests_v2_best.csv` | 19 | 10 | 74 | 103 | **all Australia** |

⇒ **"All losses are Australia" is true for both v2 arms but FALSE for v1**,
which also loses to cola_gnn and lstnet on NHS h=3 (ours 3.531 vs 2.347 /
2.566) — exactly the cell where the v2 log-growth model wins 4/5, a clean
demonstration of E2.

### Ties-only / underpowered

Entirely ties: **`japan` (20/20 T, n=70) and `state360` (20/20 T, n=72)**.
Also ties-only at cell level: `nhs` h=7 and h=14, `region785` h=10 and h=15.

Ledger `:97` recommends a **power calculation stating the minimum
detectable effect**, "so ties read as 'correctly underpowered' rather than
'no difference found'." **Not done.**

### Baseline campaign status **[computed]**

`report/logs/baseline_campaign.log`: 632 `ok` lines, 132 `FAIL`.
Grid: 5 baselines × 21 cells × 5 seeds = **525 runs**.

**517 done, 8 remaining:**
- `cola_gnn.ltla_timeseries.h-14` seeds 45, 123, 1000
- `dcrnn.ltla_timeseries.h-14` seeds 42, 30, 45, 123, 1000

These 8 are exactly why `ltla_timeseries h=14` has only 3 rows in every DM
CSV. The 132 FAILs are not real: 128 are a single mass abort at
`20260812_0329` (rc=3221226091 = STATUS_DLL_INIT_FAILED, all 0–1 s) plus 4
genuine cola_gnn LTLA h=3 failures on 14 Aug — all subsequently re-run ok.
LTLA cola_gnn runs cost 16874–37049 s each.

**The log covers the LEVEL arm only.** The 115 `_lg` prediction files were
produced by a separate, unlogged campaign.

---

## 4. Calibration / probabilistic

### What exists

- `report/results/prob_metrics.csv` — 105 rows, all
  `MSAGAT-Net / with_adj.loggrowth.quant / test / ablation=none`, i.e. the
  raw quantile-head metrics of the v2 model, 21 cells × 5 seeds.
- `report/results/conformal_metrics.csv` — 450 rows, five methods:
  `raw` (105), `identity` = per-region (105), `adjacency` (105),
  `uniform` (105), `attention` (**only 30** — missing wherever the attention
  matrix was not persisted; absent entirely for state360).

### Headline numbers **[computed]**

All datasets/horizons/seeds pooled:

| method | n | mean WIS | cov50 | **cov90** | cov95 | mean \|cov90 − 0.90\| |
|---|---|---|---|---|---|---|
| raw | 105 | 172.4 | 0.4584 | **0.8919** | 0.9551 | 0.0482 |
| **identity (per-region)** | 105 | **165.6** | 0.5122 | **0.9073** | 0.9566 | **0.0192** ← best |
| adjacency | 105 | 169.0 | 0.5171 | 0.9189 | 0.9655 | 0.0244 |
| uniform | 105 | 169.3 | 0.5151 | 0.9213 | 0.9676 | 0.0262 |
| attention | 30 | 197.6 | 0.5170 | 0.9139 | 0.9611 | 0.0278 |

**Per-region conformal is best on both mean WIS and mean coverage error,
and attention-weighted conformal is worst on both** — exactly the E6 claim,
and a direct consequence of E3.

LTLA under-coverage and its fix (5-seed means):

| h | method | cov50 | **cov90** | cov95 | Σ\|cov−nominal\| | WIS |
|---|---|---|---|---|---|---|
| 3 | raw | 0.2452 | **0.7222** | 0.8877 | 0.4949 | 16.46 |
| 3 | identity | 0.5100 | **0.9161** | 0.9676 | 0.0437 | 15.73 |
| 7 | raw | 0.2416 | **0.7702** | 0.9144 | 0.4238 | 28.92 |
| 7 | identity | 0.5220 | **0.9268** | 0.9671 | 0.0659 | 27.99 |
| 14 | raw | 0.3030 | **0.7971** | 0.9081 | 0.3418 | 48.06 |
| 14 | identity | 0.5422 | **0.9244** | 0.9704 | 0.0870 | 46.97 |

Worst single seed: h=3 seed 30, cov90 = **0.5415**. Conformal repairs it to
0.9161 *and* reduces WIS.

Per dataset (cov90 / mean WIS):

| dataset | raw cov90 | identity cov90 | raw WIS | identity WIS |
|---|---|---|---|---|
| australia-covid | 0.9128 | 0.9322 | 86.49 | **54.75** |
| japan | 0.9237 | 0.9228 | 443.3 | 443.3 |
| ltla_timeseries | **0.7632** | **0.9224** | 31.15 | 30.23 |
| nhs_timeseries | 0.8612 | 0.8756 | 3.670 | 3.551 |
| region785 | **0.9539** (over) | **0.9023** | 318.7 | 309.5 |
| state360 | 0.9017 | 0.8907 | 52.08 | 50.34 |

LTLA **under**-covers and conformal fixes it; region785 **over**-covers and
conformal pulls it back. Weekly ILI is already near-nominal raw.

---

## 5. What the documents say the next paper should contain

`doc/README.md:9` — `elservier/` = **Paper B — evaluation/methodology
rewrite, rewrite pending.**

### The three-contribution spine — ledger §6, `:196-206`

> The original claim (novel spatial-attention architecture) is not
> supportable. E3 makes it indefensible, and the interpretability analysis
> has to come out with it.
>
> What is supportable, and is arguably stronger for a methods venue:
>
> 1. **A corrected evaluation protocol** that fixes a real error in a
>    benchmark family (E1).
> 2. **Target-space design as a transferable principle**, with evidence
>    across five architectures (E2).
> 3. **The first calibrated probabilistic forecaster** on these datasets (E6).
>
> Note what this changes: the paper's centre of gravity moves from
> architecture to evaluation. That is a genuine contribution and reviewers
> of evaluation papers are receptive to negative results — E3 becomes an
> asset in that framing rather than an embarrassment. It also means the
> honest DM outcome (mostly ties) stops being a problem, because the claim
> is no longer "we win".

### The fallback definition — `program.md:260-263`

> If it fails, the paper is the evaluation paper: corrected protocol (E1),
> target-space design as a transferable principle (E2), first calibrated
> probabilistic forecaster on these datasets (E6), and the attention failure
> mode (E3/E7) as a documented negative result. That paper is safe,
> defensible, and mostly already written.

Track B did fail on accuracy, so this branch is the live one.

### How to frame E3/E7 — `doc/attention-revival-summary.md:133-153`

> Frame this as a **documented failure mode with a mechanism**, supported by
> three independent lines of evidence that agree:
> 1. **Analytical** — the sparsity penalty is provably inert…
> 2. **Diagnostic** — weight decay drove `u`,`v` to **~1e-36**; row entropy
>    reached **1.0000** on LTLA…
> 3. **Behavioural** — the module degenerates to mean pooling…
>
> The fix restores the module and improves accuracy, and **nine attempts to
> improve on it all failed** — which is itself the strongest evidence that
> the diagnosis, not the patch, is the contribution.

And `:9-20`:

> **What the fix is — and is not.** `nodecay,regpre` is an improved
> *training recipe*, not a new architecture. The model is byte-identical…
> **No reviewer should be asked to read this as architectural novelty.**

### What must be stated in the protocol paragraph — `preprocessing-audit.md:49-57`

> 1. Evaluation-protocol paragraph: state the trailing/causal smoothing
>    (LTLA, NHS only; Australia raw; weekly datasets unsmoothed), train-only
>    normalization, chronological 60/20/20 split, and the rolling-origin
>    convention that input windows may span split boundaries while targets
>    never precede the split start.
> 2. Metrics on LTLA/NHS are on the smoothed scale — keep the existing
>    limitation sentence.

### Multiple-comparisons hygiene — ledger `:123`

> Selecting a config from 100 validation-scored runs is a multiple-comparisons
> machine… Protocol: ratchet on validation only, touch test once at the end,
> re-run survivors across 5 seeds with DM. **Log the number of configurations
> tried and report it.**

Counts to report: **11 configs / 13 campaigns** (attention), **9 renewal
configurations × 5 cells = 45 comparisons**.

### Open reviewer points still owed — ledger `:83-88`

- **#1 Novelty positioning** vs adaptive graph learning, dynamic graph
  transformers, bias-based attention — **untouched**.
- **#3 150 km Haversine threshold** — unjustified, untuned, no sensitivity
  analysis, **untouched**. Now critical: "If adjacency is the only
  informative term (E3), the **threshold *is* the spatial model**."
- **#8 STAN / MepoGNN excluded** — partly addressed.
- **#9 Preprocessing** — **DONE, clean**.
- Required by `adversarial-priority-check.md:84`: a dedicated related-work
  subsection on **delay-aware graph learning outside epidemiology**.
  "This is exactly the positioning AIIM said was missing; not doing it
  guarantees another desk-level novelty rejection."

---

## 6. Contradictions between documents

### C1 — The `no_agam` sign flip, and a third data point fitting neither

| source | space | split | result |
|---|---|---|---|
| `ledger:15` (April runs) | level | test | removing EAGAM **improves** 7.95% |
| `ledger:25` | v2 log-growth | validation, seed 42 | removing EAGAM **worse 5/5, +27.1%** |
| **`aggregated_multiseed_results.csv` [computed]** | **level** | **test**, Japan only | no_agam **worse** at every horizon: h3 1236.4 vs 1133.9; h7 1424.4 vs 1388.3; h14 1475.1 vs 1457.5 |

Rows 1 and 2 reconcile mechanically. **Row 3 is level-space *and*
test-split — the same setting as row 1 — with the opposite sign.** The
−7.95% figure has **no traceable artefact**; grep finds it only in prose.
Two further hazards in that file: `none` rows have 16/24/23 seeds against 4
per ablation, and every 4-seed row lists **seed 5 twice**.

**This matters because E3 is the load-bearing negative result of the whole
evaluation paper. It must be re-derived from persisted predictions before
Paper B cites it.**

### C2 — Ledger E5 is stale for the v2-best arm

Ledger says 17/10/73; the current CSV gives **19/10/74 over 103 rows**. Any
Paper B table copying E5 verbatim will be wrong.

### C3 — "All losses are Australia" is false for v1

`dm_tests_v1_best.csv` has 2 significant losses on **nhs h=3**.

### C4 — E6's "0.433 → 0.026" does not reproduce

Nearest: LTLA h=7 mean Σ|cov−nominal| = 0.4238 → 0.0659. The cov90 pair
(0.72 → 0.92) reproduces exactly. **Restate E6 in cov90 terms.**

### C5 — `attention-revival-summary.md` never updated with its own outcome

`doc/README.md:16` says the file contains the 5-seed confirmation ("did not
generalise"); the file itself says at `:12-13` it "is running";
`ledger:48-50` says the same. **Three canonical documents disagree.**

### C6 — GraphWaveNet: claimed done, artefact says otherwise

`ledger:87` and `archive/paper_revision_recommendations.md:28-42` assert
GraphWaveNet is running on all 6 datasets.
`report/results/epilearn_baselines/epilearn_baselines_results.csv` has **one
row** (japan, h=3, seed 42, March, pre-fix, MAPE degenerate at 1.03e14), and
GraphWaveNet appears in **zero** DM comparisons.

### C7 — Data provenance does not reconcile

`ledger:125`: Spain-COVID appears as **17 regions** and as **52 regions /
122 timesteps**; LTLA appears as **307** and as **372**. One document
describes 7 datasets, the AIIM submission uses 6, Spain does not appear in
the submission at all. "Until a single canonical dataset table exists,
cross-document claims cannot be checked." *(Largely resolved by
[06-data-provenance.md](06-data-provenance.md).)*

### C8 — Split ratio drift

`preprocessing-audit.md:53` and `main.tex`: **60/20/20**.
`archive/full_comparison_analysis.md:7`, `archive/Baseline_Comparison_Results.md:19`,
`EpiSIG-Net-v3-Design.md:227`: **50/20/30**. Everything in `archive/` uses
the old ratio; comparisons across that boundary are invalid on this ground
alone, independently of E1.

### C9 — v1/v2 DM verdicts diverge because baselines' best arms differ

Any table mixing v1/v2 rows needs the arm column shown (`dm_test.py:230`).

### C10 — The proxy-grid bar vs the reported 5-seed numbers

`program.md:93-104` fixes the exp-1 bar at NHS h3 2.8082, h7 7.1788, h14
16.3110, Japan h3 504.7393, h5 635.4625 (validation, seed 42).
`tables/arms_validation.tex` reports the direct decoder over 5 seeds as h=3
**3.37 ± 0.71**, h=7 **8.56 ± 0.85**, h=14 **17.24 ± 2.44** — substantially
worse. **No seed-42 proxy-grid percentage in either track has been
re-verified at n=5.** All the "+12.5%", "+27.1%", "−6.9%" figures are
single-seed validation.

### C11 — Track B's own reversal on which formulation is correct

`renewal-net-design.md:113-122` argues the iterated version is "the faithful
version… the only route on which the original novelty claim survives";
`:234-239` reports it is **consistently worse** and calls that "a reportable
finding… not a defect." Both stand in the same file.

### C12 — Root `README.md` entirely un-updated

Still describes EAGAM as doing "scaled dot-product softmax attention…
additive structural bias" with "self-attenuating" adjacency (`:19-22` — the
exact claim E3 reverses), asserts "O(N) linear complexity while maintaining
state-of-the-art accuracy" (`:7`), lists no probabilistic metrics, and cites
AIIM 2026 as under review.

### C13 — Unresolved, nobody has raised (ledger §5, `:119-127`)

- **Scope of the E1 claim** — own harness vs the published protocol.
- **Co-author communication** (`:122`) — Palade, He, Wark, Mousavi,
  Mukandavire signed off on the AIIM version. "They need to hear this from
  you, early, framed as an audit you ran and a correction you found — not
  discovered later in a revised draft."
- **Venue** (`:124`) — "Worth a fresh venue scan."
- **Australia underperformance** (`:95`) — "All DM losses are Australia,
  both arms. **No diagnosis attempted.** Either explain it or exclude the
  dataset with justification." **[computed] confirmed:** 10/10 v2-best
  losses are Australia; at h=7 MSAGAT loses to all five baselines (ours
  410.1 vs 121.7–341.2).

---

## 7. EpiSIG-Net v3 — what it was, why parked

### What it was

`doc/EpiSIG-Net-v3-Design.md`. Positioning (`:5-9`): "the optimal model
combining efficiency and performance" — **18K parameters** (55% smaller than
v1's 40K, 14% smaller than MSAGAT-Net's 21K). "**Novel contribution**:
Serial Interval Graph (epidemiological propagation delays)."

The SIG operator (`:159-167`): `Influence_ij(t) = Σ_τ α_τ · x_j(t−τ) · A_ij`,
τ from 0 to max_lag, α_τ a learnable delay weight interpreted as the
generation-interval distribution. `MAX_LAG = 7`.

Reported results (single seed = 5): Australia h=7 RMSE **358.27** vs MSAGAT
399.86 vs v1 487.36; Spain h=7 **163.42** vs 182.47.
10-seed follow-up (`archive/Final_Model_Comparison.md:19-107`): total wins
**v1 9, MSAGAT 6, v3 5**.

The design doc's own ablation plan (`:230-241`) was **never executed**.

### Why parked — four reasons

1. **Every number predates the protocol fix** (`ledger:112`): "The 10-seed
   comparison, the Spain wins, the 'vs paper baselines' claims — none are
   currently trustworthy."
2. **It may carry the same E3 defect, never checked** (`ledger:113`): "It
   uses low-rank attention in the same codebase under the same optimiser
   configuration… Check before building on it." **Unactioned.**
3. **The novelty claim is dead on priority grounds** (D15) — the SIG is a
   learned-kernel version of a 15-year-old identity, pre-empted spatially by
   Pasetto PNAS 2023 and, for delay-in-the-graph, by PDFormer.
   Architecturally α_τ "is functionally a **learnable 1-D depthwise
   convolution over the lag axis**."
4. **Track B absorbed its only defensible content** — the learned generation
   interval — into the renewal decoder on the MSAGAT backbone, making it a
   controlled decoder comparison. Track B closed 0/45 on accuracy but earned
   the interpretability result, so the SIG's residual contribution has
   effectively been tested and reported.

### What is still attractive — ledger `:101-116`

The SIG is described as "**the strongest novelty claim anywhere in this
corpus**" because it is epidemiologically grounded, cleanly differentiated,
supported by the project's own data — adjacent-region correlation decays
**0.960 (lag 0) → 0.877 (lag 1) → 0.560 (lag 3) → 0.050 (lag 7)** — and
**independent of the broken attention path**: "SIG operates on lagged
signals; it does not depend on EAGAM carrying signal. **E3 does not touch
it**" (`:108`).

---

## Appendix — artefact index

| Path | What it is |
|---|---|
| `report/results/dm_tests_v2_best.csv` | 103 rows, v2 vs best baseline arm — **19W/10L/74T** |
| `report/results/dm_tests_v2_level.csv` | 101 rows, v2 vs published level arm — 19W/8L/74T |
| `report/results/dm_tests_v1_best.csv` | 101 rows, v1 vs best arm — 15W/7L/79T |
| `report/results/conformal_metrics.csv` | 450 rows; 5 methods × 21 cells × 5 seeds (attention incomplete, 30/105) |
| `report/results/prob_metrics.csv` | 105 rows; the `raw` slice, v2 test split |
| `report/results/attn_revival_ledger.csv` | 26 rows × 17 cols; every Track A/B config |
| `report/results/epiestim_validation.csv` | 5 rows: japan h3/h5, nhs h3/h7/h14 |
| `report/results/aggregated_multiseed_results.csv` | **stale (Feb 2026)**, Japan-only, uneven seed counts |
| `report/results/epilearn_baselines/…csv` | **1 row**: GraphWaveNet japan h3 seed 42 |
| `report/logs/baseline_campaign.log` | 517/525 cells done; 8 remaining |
| `report/predictions/*/` | 566 MSAGAT npz across 31 variant tags; 624 baseline npz incl. 115 `_lg` |
| `doc/plos-renewal/` | Paper A: 695-line `main.tex`, 6 tables, 4 figures |
| `doc/elservier/` | Paper B bundle — rewrite pending |
| `src/scripts/dm_test.py` | the DM harness; `program.md:201` forbids modifying it |
