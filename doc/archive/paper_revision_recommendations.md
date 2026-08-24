# MSAGAT-Net Paper Revision Recommendations

Prioritised list of actions to strengthen the paper for Q1 submission.

**Target journal:** Artificial Intelligence in Medicine (AIM) — Q1, IF ~7.5, Elsevier hybrid, FREE OA via Coventry JISC agreement.
**Fallback:** Computers in Biology and Medicine — Q1, IF ~7.0, also hybrid/free.
**Verify eligibility:** agreements.journals.elsevier.com/jisc | Contact: oa.lib@coventry.ac.uk

Items are ordered by expected reviewer impact.

**Field context on statistical testing (from checking baselines):**
- EpiGNN (ECML PKDD 2022): runs 5 seeds but reports only point estimates, no std in tables
- Cola-GNN (CIKM 2020): 10 trials, reports mean +/- std in figures (most rigorous baseline)
- DCRNN (ICLR 2018): single-run, no multi-seed
- STAN (JAMIA 2021): bootstrap resampling with 95% CI and t-tests (medical venue, higher bar)
- LSTNet (SIGIR 2018): single-run, no multi-seed

Single-seed is the de facto norm in CS venues for this field. AIM is a medical informatics journal (closer to JAMIA), so reviewers may hold a higher standard, but single-seed alone should not cause rejection given that 3 of 5 baselines do the same.

---

## Priority 1: Most likely to cause a "reject" or "major revision"

### 1.1 Add recent graph-based baselines

**Problem:** The paper discusses 8+ recent epidemic GNNs in Related Work but only compares against 5 baselines, two of which (LSTNet, CNNRNN-Res) don't model spatial structure.

**Completed: GraphWaveNet added via EpiLearn.** GraphWaveNet (Wu et al., 2019) is the strongest addition because it learns adaptive adjacency from data via node embeddings — directly comparable to MSAGAT-Net's learned graph bias B=UV. Experiments running on all 6 datasets.

**Other models investigated but not viable:**
- **STAN** (Gao et al., 2021) — physics-informed (SIR loss), requires recovered counts + population data we don't have. Exclude with justification.
- **MepoGNN** (Cao et al., 2022) — physics-informed (SEIR + OD mobility data). Same issue.
- **DASTGN** (Pu et al., 2024) — EpiLearn implementation hangs with real adjacency matrices (confirmed bug).
- **GraphLSTM** — EpiLearn implementation crashes (propagate() missing x argument).
- **HierST** — Empty module in EpiLearn (not implemented).
- **MSGNN** (Qiu et al., 2024) — not in EpiLearn, no public code found.

**Paper justification for exclusions:** "Physics-informed models (STAN, MepoGNN) require compartmental state data and external data sources unavailable for all datasets; we restrict comparisons to purely data-driven approaches."

**Where in paper:** Add GraphWaveNet to Tables 2 and 3. Add to Section 4.5. Update results discussion.

**Effort:** Done (GraphWaveNet running). Paper edits needed.

### 1.2 Ablation on multiple datasets

**Problem:** The ablation study is only on Japan-Prefectures (47 nodes). The horizon-dependent patterns claimed may not generalise. This is a common reviewer complaint: "How do I know this isn't dataset-specific?"

Key questions the ablation should answer:
- Does EAGAM still dominate on LTLA (372 nodes, dense graph)?
- Does MSSFM contribute more on large graphs where multi-hop should matter?
- Does MSSFM contribute on Australia (8 nodes) where oversmoothing is the risk?

**Action:**
- Run the 3 ablation variants (no_agam, no_mtfm, no_pprm) on at least 2 more datasets: LTLA-Timeseries (large graph) and one small-graph dataset (Australia-COVID or NHS-Timeseries).
- If MSSFM still shows minimal contribution on large graphs, discuss honestly rather than overselling.

**Where in paper:** Add a second ablation table or extend Table 4. Update ablation discussion in Section 5.2.

**Effort:** Medium. Ablation code already exists. Needs compute time only.

---

## Priority 2: Strongly recommended (strengthens the paper significantly)

### 2.1 Multi-seed evaluation

**Problem:** All results are single-seed. While this is consistent with DCRNN, LSTNet, and EpiGNN's reporting, AIM as a medical informatics venue may expect more rigour (STAN at JAMIA reported confidence intervals). The "up to 23.5% improvement" claim would be stronger with variance estimates.

**Recommended approach (tiered):**
- **Minimum:** Run MSAGAT-Net only with 5 seeds on all 6 datasets. Report mean +/- std. This alone exceeds what EpiGNN, DCRNN, and LSTNet provide. Baselines keep single-seed values (justified by citing that their original papers also use single-seed or fixed seeds).
- **Better:** Run MSAGAT-Net and the 3 graph-based baselines (DCRNN, Cola-GNN, EpiGNN) with 5 seeds. Skip LSTNet and CNNRNN-Res (non-graph, lower priority).
- **Best:** All models, 5 seeds, add Wilcoxon signed-rank test or paired t-test for the top comparisons.

**Where in paper:** Update Tables 2 and 3. Add a sentence in Section 4.4 about the protocol. If you do multi-seed, remove the limitation sentence in the Conclusion; if not, keep it.

**Effort:** High for "best", Low-Medium for "minimum". The minimum approach (MSAGAT-Net only) would take ~1 day of compute.

### 2.2 Empirical evidence for self-regulating property

**Problem:** The "self-regulating" claim is currently analytical only. A reviewer may say: "The math makes sense, but show me this actually happens on your datasets."

**Action (pick 1-2, in order of impact):**
1. Report the learned `adj_scale` (alpha) values after training on each of the 6 datasets. If alpha is larger on sparse graphs and smaller on dense ones, this directly validates the claim. This is a one-line extraction from each checkpoint — very low effort.
2. Compute attention entropy with and without the adjacency prior across datasets. Show that on dense graphs (Australia, 8 fully-connected nodes), the prior barely shifts the entropy, while on sparse graphs (Japan, LTLA), it shifts it meaningfully.
3. Side-by-side attention heatmaps for a dense vs. sparse graph.

**Where in paper:** Add a small table (6 rows, one per dataset) showing: dataset, N nodes, graph density, learned alpha, attention entropy with/without prior. One paragraph of discussion. Could go in Section 5.2 near the attention visualisations.

**Effort:** Low. Models already trained; just extract values.

### 2.3 Strengthen adjacency-free validation

**Problem:** The paper claims MSAGAT-Net doesn't need adjacency, but only tests this on Japan h=10. MSSFM becomes identity aggregation without adjacency, so only EAGAM's learned graph bias provides spatial modelling.

**Action:**
- Run MSAGAT-Net without adjacency on at least 3 datasets at multiple horizons.
- Report in a small table: dataset, horizon, RMSE with adj, RMSE without adj, delta.
- Be explicit that without adjacency, MSSFM defaults to identity and spatial learning relies entirely on EAGAM's low-rank graph bias B = UV.

**Where in paper:** Expand the adjacency-free discussion at the end of Section 5.2.

**Effort:** Low. Re-run existing models with `adj_matrix=None`.

---

## Priority 3: Would strengthen but not critical

### 3.1 Parameter count and inference time comparison

**Problem:** The paper mentions "parameter efficiency" but never reports numbers.

**Action:** Add a small table: model name, parameter count, training time per epoch, inference time per sample. MSAGAT-Net has ~21K parameters. This is a concrete differentiator.

**Where in paper:** Section 4.5 or a new efficiency subsection in Results.

**Effort:** Low (1-2 hours).

### 3.2 Discuss limitations of the adaptive hop heuristic

**Problem:** `S = min(S_max, max(2, N//5))` is hand-crafted. A reviewer may ask why N/5.

**Action:** Either (a) add a sensitivity analysis in an appendix testing S_max = 2, 3, 4, 6 on one dataset, or (b) be transparent that this is a practical heuristic motivated by the oversmoothing literature, not an optimised hyperparameter.

**Effort:** Low.

### 3.3 MSSFM contribution is weak — prepare a response

**Problem:** Removing MSSFM improves RMSE at 3-day (-1.34%) and barely degrades at 7-day (+1.71%) and 14-day (+1.19%) on Japan. A reviewer may question whether MSSFM contributes.

**Possible responses:**
- Multi-seed ablation (2.1) would show whether these differences are consistent or within noise.
- Ablation on LTLA (372 nodes) where multi-hop aggregation should matter more may show a stronger MSSFM contribution.
- If MSSFM genuinely doesn't help much, frame it honestly: "MSSFM provides a modest contribution on Japan-Prefectures but its utility increases with graph size" (if LTLA ablation supports this). Honest findings about when components don't help are valued by reviewers.

**Effort:** Depends on 1.2 and 2.1.

---

## Suggested execution order

If time-limited, work in this order:
1. **2.2** (extract alpha values) — 1 hour, immediate novelty boost
2. **1.2** (ablation on LTLA + Australia) — 1 day compute, addresses two concerns at once
3. **2.3** (adj-free runs) — can run alongside 1.2
4. **1.1** (add STAN baseline) — 3-5 days including code adaptation
5. **2.1 minimum** (MSAGAT-Net multi-seed) — 1 day compute
6. **3.1** (parameter counts) — 1 hour

---

## Already completed (in previous editing session)

- [x] Fixed abstract factual error (removed "dilated multi-scale temporal features")
- [x] Fixed broken citation keys (deng2020cola, li2018diffusion)
- [x] Removed misleading "quadratic attention efficiency" claim
- [x] Trimmed verbose multi-head attention equations
- [x] Removed redundant PPRM text
- [x] Tightened dataset and graph construction prose
- [x] Removed all content em-dashes
- [x] Tightened Learned Spatial Representations and Conclusion
- [x] Added oversmoothing citation (Li et al., 2018)
- [x] Added ALiBi and Shaw citations for explicit differentiation
- [x] Reframed novelty: honest about building on established techniques, clear about what the specific integration contributes
- [x] Added explicit contrast with how Cola-GNN and EpiGNN inject structure
- [x] Added single-seed limitation acknowledgment
- [x] Added fixed-seed reproducibility note
