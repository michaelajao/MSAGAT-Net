# 03 — Baselines, follow-on work, and where MSAGAT-Net sits

Literature read 24 Aug 2026 from primary sources (arXiv, publisher pages,
author copies). Every claim carries a URL. Two gaps could not be closed and
are flagged at the end.

**Bottom line.** The three components MSAGAT-Net claims — multi-scale
*dilated* temporal convolution, spatial attention *gated against a
geographic adjacency*, and *graph message-passing propagation* — are not
merely precedented: they are **the three modules of Cola-GNN itself** (2020),
re-used wholesale by EpiGNN (2022). A Knowledge-Based Systems paper from
**April 2026** (STTGNN) publishes essentially the same architecture on two of
the three benchmark datasets. Separately, two 2026 benchmark papers show the
whole *dataset family* is compromised as an accuracy arena.

---

## Part A — What the baselines actually proposed

### A1. Cola-GNN (Deng, Wang, Rangwala, Wang, Ning; CIKM 2020)

Author copy: <https://yue-ning.github.io/docs/CIKM20-colagnn.pdf> ·
DOI <https://doi.org/10.1145/3340531.3411975>

**Core mechanism — the crux of the problem.** Cola-GNN has exactly three
modules (§3.2–3.4):

1. **§3.2 Directed Spatial Influence Learning (location-aware attention).**
   A global RNN produces per-location hidden states `h_i`; additive attention
   gives an *asymmetric* coefficient
   `a_{i,j} = vᵀ g(Wˢh_i + Wᵗh_j + bˢ) + bᵛ` (Eq. 2), row-normalised by
   max-ℓp norm (Eq. 3). Then — this is the "learnable adjacency bias":

   > Ã^g = D^(−1/2) A^g D^(−1/2)  (Eq. 4)
   > **M** = σ(W^m **A** + b^m 1_N 1_Nᵀ)  (Eq. 5)
   > **Â** = **M** ⊙ Ã^g + (1_N 1_Nᵀ − **M**) ⊙ **A**  (Eq. 6)

   i.e. a *learned element-wise gate* **M** blending the normalised
   geographic adjacency with the learned attention matrix. The paper calls it
   "adapted from the feature fusion gate… dynamically learned, and it weighs
   the contribution of geographic and historical information".

2. **§3.3 Multi-Scale Dilated Convolution.** Verbatim heading. "we adapt a
   multi-scale dilated convolutional module which consists of multiple
   parallel convolutional layers with the same filter and stride size but
   different dilation rates… For short-term and long-term patterns, we define
   K filters with dilation rates k_s and k_l (k_l > k_s)." K = 10, k_s = 1,
   k_l = 2.

3. **§3.4 Graph Message Passing – Propagation.**
   `h_i^(l) = g(Σ_{j∈N} â_{i,j} W^(l−1) h_j^(l−1) + b^(l−1))`, initialised
   from the dilated-conv features, 2 graph layers.

Their own ablation (Table 4) is literally named **`Cola-GNN w/o temp`**
(remove the dilated temporal conv), **`w/o loc`** (remove location-aware
attention), **`w/o geo`** (remove the geographic adjacency from Eqs. 4–6).

**Protocol.** Japan-Prefectures 47×348 (IDWR, Aug 2012–Mar 2019),
US-Regions 10×785 (ILINet/HHS, 2002–2017), US-States 49×360 (CDC, 2010–2017,
Florida dropped). Split **50% train / 20% val / 30% test**, chronological;
min–max normalised per location using training data. **Window T = 20 weeks.**
Lead times **h ∈ {2, 3, 5, 10, 15}** (h=1 excluded "because symptom
monitoring data is usually delayed by at least one week"). Metrics **RMSE,
MAE, PCC**. **"All experimental results are the average of 10 randomized
trials."** Adam, lr ∈ {0.001, 0.005, 0.01}, batch 32, dropout 0.2, ELU.
Separate model per horizon.

**Reported RMSE (Table 3), h = 2 / 3 / 5 / 10 / 15:**

| Model | Japan-Prefectures | US-Regions | US-States |
|---|---|---|---|
| **Cola-GNN** | **929 / 1051 / 1117 / 1372 / 1475** | **480 / 636 / 855 / 1134 / 1203** | **136 / 167 / 202 / 241 / 237** |
| ST-GCN | 996 / 1115 / 1129 / 1541 / 1527 | 697 / 807 / 1038 / 1290 / 1286 | 189 / 209 / 256 / 289 / 292 |
| CNNRNN-Res | 1133 / 1550 / 1942 / 1865 / 1862 | 571 / 738 / 936 / 1233 / 1285 | 205 / 239 / 267 / 260 / 250 |
| LSTNet | 1133 / 1459 / 1883 / 1811 / 1884 | 554 / 801 / 998 / 1157 / 1231 | 199 / 249 / 299 / 292 / 292 |
| DCRNN | 1502 / 1769 / 2024 / 2019 / 1992 | 711 / 874 / 1127 / 1411 / 1434 | 165 / 209 / 244 / 299 / 298 |
| LSTM | 1052 / 1246 / 1335 / 1622 / 1649 | 507 / 688 / 975 / 1351 / 1477 | 150 / 180 / 213 / 276 / 307 |

Cola-GNN PCC: Japan 0.915/0.901/0.890/0.813/0.753 · US-Regions
0.946/0.909/0.835/0.717/0.639 · US-States 0.955/0.933/0.897/0.822/0.856.

Model size: **Cola-GNN = 3K parameters**, 0.21 s/epoch (Table 5). They warn:
"deep learning-based models with high model complexity tend to overfit due to
the small size of training data in the epidemic domain" and "More filters
tend to reduce the predictive power of the model".

**Future work, §6, verbatim:**

> "One shortcoming of the proposed method is training flexibility. Separate
> models are trained for different lead time settings. In the future, we will
> consider iterative predictions to increase model flexibility. Another
> research direction is to involve more complex dependencies such as social
> factors, climate changes, and population migration. We intend to determine
> if the prediction accuracy is improved when using external indicators.
> Furthermore, it is also essential to identify the main factors affecting the
> epidemic outbreak of one location by learning multiple locations
> simultaneously."

### A2. EpiGNN (Xie, Zhang, Li, Zhou, Tan; ECML-PKDD 2022)

<https://arxiv.org/abs/2208.11517> · <https://doi.org/10.1007/978-3-031-26422-1_29> ·
code <https://github.com/Xiefeng69/EpiGNN>

- **§3.2 Multi-Scale Convolutions** — "we also adopt multi-scale convolutions
  with different filter sizes and **dilated factors** as a feature
  extractor." Five filters: `{f_{1×3,1}, f_{1×5,1}, f_{1×3,2}, f_{1×5,2},
  f_{1×T,1}}`.
- **§3.3 Transmission Risk Encoding.** *Local* (LTR):
  `h_i^l = W^l · d_i + b^l` where `d_i` is the node's **degree in the
  geographic graph**. *Global* (GTR): a self-attention correlation matrix,
  row-normalised, row-summed to `g_i`, then `h_i^g = W^g g_i + b^g`.
- **§3.4 Region-Aware Graph Learner.** Node features
  `h^feat = h^temp + h^l + h^g`. Asymmetric temporal graph
  `Â = ReLU(tanh(M₁M₂ᵀ − M₂M₁ᵀ))` (Eq. 9). Then the **degree-gated adjacency
  bias**:

  > **D**ˢ = sigmoid(**W**ˢ ∘ **dd**ᵀ)  (Eq. 10)
  > **Ā** = **D**ˢ ∘ **A**^geo + **Â**  (Eq. 11)

  where **W**ˢ ∈ ℝ^{N×N} is "a learnable parameter matrix".
- **§3.5** 1–5 layer GCN; **§3.6** optional linear **autoregressive** branch.

**Protocol.** Same 50/20/30 split, **T = 20**, batch 128, **5 runs**, metrics
**RMSE and PCC only**. Horizons {3,5,10,15} for flu, {3,7,14} for COVID.
Five datasets: the three Cola-GNN flu sets plus **Australia-COVID (8 regions
× 556 days, JHU-CSSE, 27 Jan 2020 – 4 Aug 2021)** and Spain-COVID.

**Reported RMSE / PCC (Table 2), h = 3 / 5 / 10 / 15:**

| Model | Japan-Prefectures | US-Regions | US-States |
|---|---|---|---|
| **EpiGNN RMSE** | **996 / 1031 / 1441 / 1470** | **589 / 774 / 984 / 1061** | **160 / 186 / 220 / 236** |
| **EpiGNN PCC** | 0.904 / 0.908 / 0.739 / 0.773 | 0.912 / 0.842 / 0.749 / 0.694 | 0.935 / 0.907 / 0.865 / 0.861 |
| Cola-GNN* RMSE | 1051 / 1117 / 1372 / 1475 | 636 / 855 / 1134 / 1203 | 167 / 202 / 241 / 237 |

`Cola-GNN*` = copied from the CIKM paper, marked with an asterisk — they did
**not** re-run it. EpiGNN **loses** to Cola-GNN at Japan h=10 (1441 vs 1372).
The headline "9.48% RMSE" is against *the best baseline per cell*; §4.2 gives
the honest figure: "EpiGNN achieves **5.6%** and 13.4% lower RMSE than the
best baselines in the influenza prediction task and COVID-19 prediction task
respectively."

Australia-COVID (Table 3, RMSE h=3/7/14): EpiGNN **71.42 / 153.07 / 287.90**;
Cola-GNN 127.59 / 279.56 / 326.79; AR 85.21 / 237.73 / 309.03. Sizes: EpiGNN
9–12K params vs Cola-GNN 7–9K.

**Future work, §5, verbatim (one sentence, that is all there is):**

> "As for future work, we will devote to better predict by considering the
> time decay effects of spatial transmission."

### A3. MepoGNN (Cao et al.; ECML-PKDD 2022)

<https://arxiv.org/abs/2306.14857> · <https://doi.org/10.1007/978-3-031-26422-1_28>

Embeds a **metapopulation SIR** into a GNN so that time- and region-varying
β, γ and the propagation graph are learned end-to-end from mobility (adaptive
and dynamic variants). Japan COVID-19, 47 prefectures + daily mobility flow.
**Not on the benchmark** — it needs OD mobility flows, which the Cola-GNN
datasets do not have.

### A4. STAN (Gao, Sharma, Cui, Xiao, Malin, Sun; JAMIA 28(4):733–743, 2021)

<https://academic.oup.com/jamia/article/28/4/733/6118380> ·
<https://doi.org/10.1093/jamia/ocaa322>

**GAT over a location graph** (edges weighted by population product and
geographic distance) → **GRU** → dual-head output, trained with a
**transmission-dynamics constraint loss**: the network predicts β and γ, an
SIR/SEIR ODE is rolled forward, and the discrepancy is penalised. Data: JHU
COVID cases 22 Mar–10 Jun 2020 + **IQVIA claims** (48 COVID diagnosis codes),
45 states / 193 counties.

**Limitations, verbatim:**

> "If the number of cases fluctuates drastically due to inaccuracy in the
> data collection process, it is difficult for the STAN model to learn valid
> and stable transmission and recovery rates."
> "The transmission dynamics constraints may be too simple to reflect
> real-world situations, such as home isolation and pandemic control policies."

Full analysis of STAN's mechanism, data requirements and protocol holes is in
[04-user-papers-folder.md](04-user-papers-folder.md).

### A5. CNNRNN-Res — correction to the citation

It is **SIGIR 2018, not KDD 2018**: Yuexin Wu, Yiming Yang, Hiroshi Nishiura,
Masaya Saitoh, *"Deep Learning for Epidemiological Predictions"*, SIGIR '18,
pp. 1085–1088. <https://dl.acm.org/doi/10.1145/3209978.3210077>

RNN for long-range temporal correlation + CNN to **fuse signals across data
sources/locations** + **residual links**. Cola-GNN's commentary: "CNNRNN-Res
uses geographic location information and they only perform well on the
US-States and Japan-Prefectures datasets, respectively."
**Fix this citation before resubmission — a reviewer who checks will not be
charitable.**

### A6. DCRNN and LSTNet — general models, not epidemic models

- **DCRNN** — Li, Yu, Shahabi, Liu, ICLR 2018.
  <https://arxiv.org/abs/1707.01926>. Traffic as a **diffusion process on a
  directed graph** inside a GRU encoder–decoder with scheduled sampling.
  Designed for dense 5-minute sensor networks; on 348–785-point weekly ILI it
  is badly over-parameterised, which is why it is the worst deep baseline in
  Cola-GNN's Table 3. Cola-GNN: "The DCRNN model performs very unstably on
  these three data sets, especially in long-term settings."
- **LSTNet** — Lai, Chang, Yang, Liu, SIGIR 2018.
  <https://arxiv.org/abs/1703.07015>. CNN + **recurrent-skip** +
  **autoregressive** linear branch. **Graph-free.** Cola-GNN: "the complexity
  of these models is very high, leading to overfitting on the flu prediction
  task."

Framing both as "epidemic baselines" without saying they are off-the-shelf
traffic/MTS models tuned for a different data regime is a weakness reviewers
notice.

---

## Part B — What came after (2023–2026)

### B1. The field's own agenda: KDD 2024 survey

**Liu, Wan, Prakash, Lau, Jin — "A Review of Graph Neural Networks in
Epidemic Modeling", KDD 2024.** <https://arxiv.org/abs/2403.19852> ·
paper list <https://github.com/Emory-Melody/awesome-epidemic-modeling-papers>

Section 5 ("Future Work") has **six** headings. Verbatim opening sentences:

1. **Epidemic at Scales** — "Multi-scale data are crucial in epidemiology
   because they offer comprehensive insights into both intra-region and
   inter-region relationships…" Existing methods handle "only two predefined
   scales, such as county-level and state-level data".
2. **Cross-Modality in Epidemiology** — "The integration of multi-modal data
   in epidemiological tasks offers a powerful approach…"
3. **Epidemic Diffusion Process** — "all GNN-based methods discussed above
   involve information aggregation at one or several time points in a
   **discrete** manner"; they call for **continuous** GNNs and handling of
   asynchronous transmission timing.
4. **Interventions for Epidemics** — current work uses "only one type of
   intervention, either node-level or edge-level".
5. **Generating Explainable Predictions** — "neural models investigated thus
   far have not placed significant emphasis on this aspect."
6. **Handling Challenges from Epidemic Data** — noisy, incomplete, and
   **private** data, with **Federated Graph Learning** flagged as promising.

**Note what is NOT on this list: "a better attention mechanism", "a better
adjacency prior", or "multi-scale temporal convolution".** The field's own
agenda has moved past architecture tweaking — precisely the gap AIIM read the
manuscript against.

Companion survey: Rodríguez et al., *"Data-Centric Epidemic Forecasting: A
Survey"* — <https://arxiv.org/abs/2207.09370>

### B2. Benchmark and evaluation papers — the most important development

#### SpatialEpiBench (Lyu, Turcan, Wilder; CMU; arXiv 2605.06530, 7 May 2026)

<https://arxiv.org/abs/2605.06530>

11 datasets (ILINet, NCHS deaths, CHNG in/outpatient, CPR admissions, DV
doctor visits, HHS hosp, JHU cases, Canada, **Australia**), **rolling-origin
retraining** ("Models are retrained from scratch every 8 time steps using the
most recent 100 observations available at that origin"), plus
outbreak-specific metrics. Models: DCRNN, AGCRN, STGCN, GraphWaveNet, MTGNN,
GTS, StemGNN, STNorm, **EpiGNN, Cola-GNN, EARTH**, vs DLinear, ARIMA,
**Naive persistence**.

Findings, effectively verbatim:

- **"Every method beats the naive baseline less than 50% of the time."**
- **"Most methods underperform naive, and adjacency-informed methods do not
  beat univariate baselines."**
- Even during outbreaks, **"almost no method outperforms naive."**
- Three failure modes: **(1) poor outbreak anticipation; (2) difficulty
  handling sparsity and noise; (3) limited utility of common geographic
  adjacency for epidemiological spatial information.**
- On prior practice: existing evaluations are **"heterogeneous, narrow, and
  often misaligned with practice"**; "current evaluations often used simple
  chronological train-test splits that do not reflect real-time forecasting
  practice"; "Prior work (Cola-GNN, EpiGNN, EpiColaGNN, EARTH) tested on 1–5
  datasets without rolling evaluation or outbreak metrics."

**This is a direct, named, peer-visible refutation of the value proposition of
a Cola-GNN-family architecture paper, published three months before the AIIM
rejection.**

#### EpiCastBench (Panja, D'Agostino, Li, Chakraborty, Liu; arXiv 2605.11598, 12 May 2026)

<https://arxiv.org/abs/2605.11598> · <https://github.com/aimltsf/EpiCastBench>

40 multivariate epidemic datasets, 8 diseases, 15 models, rolling windows,
**Friedman test + Nemenyi/MCB post-hoc significance testing** (p<0.01).
Winners: **foundation models — TimesFM (long horizon), Chronos-2
(short/medium)**. Tree/linear models "occasionally outperform more complex
frameworks". **GNNs were not included** — named as a gap, with "extension to
spatiotemporal settings", probabilistic forecasting and mechanistic
integration recommended.

#### "From naive to foundation" (Wang, Li, Perra; QMUL; medRxiv, May 2026)

<https://www.medrxiv.org/content/10.64898/2026.05.11.26352889v1.full>

ILI in 9 European countries, 4 seasons, rolling origin, MAE/WMAPE/IS₈₀.
**TabPFN-TS zero-shot "consistently outperforms all other individual
architectures"**; the ECDC **RespiCast ensemble** wins at 3–4 weeks. Key
caveat: "unaugmented deep learning frequently fails" to beat naive.

#### Operational reality checks

- **Cramer et al., PNAS 119(15) e2113561119, 2022** —
  <https://www.pnas.org/doi/10.1073/pnas.2113561119>. US COVID-19 Forecast
  Hub, 90+ teams: the ensemble "exceeded the performance of all of the models
  that contributed to it."
- **Mathis et al., Nature Communications 15:6289, 2024** —
  <https://www.nature.com/articles/s41467-024-50601-9>. FluSight,
  WIS/relative-WIS/coverage: **only 6 of 23 models beat the baseline in
  2021–22** (12/18 in 2022–23).

Operational epidemic forecasting is **probabilistic and ensemble-based**.
Deterministic point-forecast RMSE/PCC leaderboards — the AIIM manuscript's
entire evaluation frame — have no counterpart in the deployed literature.

### B3. Architectural successors on the exact datasets

#### STTGNN — the single most damaging precedent

**"A multi-scale spatio-temporal transformer with region-aware graph learning
for epidemic forecasting", Knowledge-Based Systems, published 1 April 2026.**
<https://www.sciencedirect.com/science/article/abs/pii/S0950705126006374>

From the abstract: "STTGNN, a novel Spatio-Temporal Transformer Graph Neural
Network that integrates **multi-scale temporal convolutions**, a two-stage
spatio-temporal Transformer, and a **region-aware graph learner enhanced with
degree-aware gating**. A lightweight Graph Convolutional Network is employed
to **propagate signals over dynamically inferred graphs**… A region-aware
graph learner is introduced to infer **directed and time-varying adjacency
structures, which are adaptively fused with static geographic information via
a degree-aware gating mechanism**… Extensive experiments conducted on
**Japan-Prefectures and US-Regions** datasets demonstrate that STTGNN
consistently outperforms statistical baselines, deep sequence models, and
state-of-the-art spatio-temporal graph methods in terms of **RMSE, PCC, and
peak error**."

Multi-scale temporal conv + attention + learnable-gated geographic adjacency
+ GCN propagation, on our datasets, with our metrics. **That is MSAGAT-Net.**
Paywalled — the full text and results table could not be retrieved. **Obtain
this PDF before writing another word of revision.**

#### HeatGNN (Zheng, Jiang, Chen, Zhou, Zhan, Nguyen, Yin) — arXiv 2411.17372

<https://arxiv.org/abs/2411.17372>

Binds a mechanistic SIR into a GNN: five MLPs parameterise time-varying S, I,
R, β, γ under a **physics-informed loss**; a **time-varying "mechanistic
affinity" transmission graph**. Backbone: **EpiGNN**. Datasets: the three flu
sets **plus Australia-COVID**. Gains over Cola-GNN at h=2 (RMSE ×10³): Japan
1.149 vs 1.168 (**+1.6%**), US-Regions 0.541 vs 0.552 (**+1.9%**), US-States
0.142 vs 0.148 (**+4.1%**), Australia-COVID 0.315 vs 0.368 (+14.4%).

**Read those gain figures.** A well-executed mechanistic hybrid gets
**1.6–4.1%** on the flu sets. Anything reported much above that on the same
protocol will read as a protocol artefact, not a result.

#### EARTH — Epidemiology-Aware Neural ODE (Wan, Liu, Lau, Prakash, Jin)

<https://arxiv.org/abs/2410.00049> · ICML 2025

**EANO**: SIR compartments as latent variables inside a **neural ODE**.
**GLTG**: global infection indicators via **DTW**, modulating a local
transmission graph, with a mask balancing static geography against learned
dynamic edges. Datasets: US-Regions, US-States, Australia-COVID. Reports
**Peak Time Error** alongside RMSE.

| Dataset | Method | h=5 | h=10 | h=15 |
|---|---|---|---|---|
| US-States | ColaGNN | 299.1 / 81.53 | 283.4 / 79.12 | 339.4 / 120.6 |
| | EpiGNN | 288.5 / 84.32 | 297.6 / 84.32 | 391.6 / 157.4 |
| | EpiColaGNN | 286.1 / 83.38 | 300.9 / 90.65 | 375.1 / 132.5 |
| | **EARTH** | **243.2 / 67.43** | **277.8 / 80.43** | **300.1 / 104.2** |
| US-Regions | ColaGNN | 1148 / 533.6 | 1524 / 846.6 | 1552 / 856.3 |
| | EpiGNN | 1136 / 534.2 | 1454 / 728.9 | 1444 / 764.2 |
| | **EARTH** | **1080 / 522.4** | **1244 / 605.3** | **1301 / 647.1** |
| Australia-COVID | ColaGNN | 224.2 / 55.23 | 544.8 / 161.6 | 795.8 / 258.0 |
| | EpiGNN | 210.3 / 40.12 | 467.3 / 120.1 | 764.2 / 233.7 |
| | **EARTH** | **156.8 / 30.12** | **177.6 / 38.62** | **225.3 / 56.32** |

**Look at US-States h=5: EARTH reports Cola-GNN at RMSE 299.1. Cola-GNN's own
paper reports 202. EpiGNN copies 202.** The "benchmark" is not a benchmark.

#### EpiHybridGNN (Kong, Wang, Li, Chen, Lu; arXiv 2511.15469, Nov 2025)

<https://arxiv.org/abs/2511.15469>

Literally a Cola-GNN ⊕ EpiGNN merge, on the three flu sets, horizons 2–32.
Worth quoting to co-authors: EpiGNN "excels in short-term forecasting" but
shows "increasing errors in the long term", while ColaGNN has "stronger and
more consistent overall performance". **That a 2025 paper's entire
contribution is gluing our two baselines together, and it got no venue better
than arXiv, is diagnostic of how saturated this niche is.**

#### Others on or near this family

- **MSGNN** (Qiu, Tan, Bao; DMKD 38:2348, 2024) —
  <https://arxiv.org/abs/2308.15840>. Multi-scale **graph** (not multi-scale
  time). **US COVID-19 only.**
- **BDSTGNN** (Mao, Han, Tanaka, Wang; KBS 2024) —
  <https://arxiv.org/abs/2312.00485>. Static "backbone" + dynamic temporal
  graph; **DLinear replaces recurrence**.
- **M-SPICE** (Gomez, Wu, Wang, Shen, Rodríguez; **KDD 2026**) —
  <https://arxiv.org/html/2606.22171>. Multimodal spatial maps ⊕ temporal
  transformer. **ColaGNN NRMSE 0.291 vs Persistence 0.213 vs M-SPICE 0.194 —
  Cola-GNN loses to persistence.**
- **EASTG** (Xu et al., IEEE BIBM 2025) — adaptive graph + trend/seasonal
  dual-stream + **next-generation matrix**, requiring *only reported case
  counts*.
- **CSTGNN** (Han et al., 2025) — Spatio-Contact SIR ⊕ GNN.
- **EpiDHGNN** (Liu et al.; arXiv 2503.20114) — dynamic **hypergraphs**,
  individual-level data.
- **PISID** (PLOS ONE 20(9):e0331611, 2025) —
  <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0331611>.
  STID-style spatio-temporal identity embeddings **with no graph at all**,
  plus an SIR module; ~27K params, "stable and superior predictive
  performance". **A graph-free model matching graph models is itself a finding
  to engage with.**
- **HierST** (Zheng et al., KDD 2021) — hierarchical county/state/national
  consistency, deployed in the COVID-19 Forecast Hub.
- **DASTGN** (Pu et al., CAAI Trans. Intelligence Technology, 2024).
- **MPNN+LSTM / MPNN+TL** (Panagopoulos, Nikolentzos, Vazirgiannis, AAAI
  2021) — <https://ojs.aaai.org/index.php/AAAI/article/view/16616>. Still one
  of the few serious attempts at cross-outbreak generalisation. Full analysis
  in [04-user-papers-folder.md](04-user-papers-folder.md).
- **CAMul** (Kamarthi et al., WWW 2022) — <https://arxiv.org/abs/2109.07438>.
  Probabilistic **multi-view** forecasting, ">25% in accuracy and calibration".
- **EINNs** (Rodríguez et al., AAAI 2023) —
  <https://arxiv.org/abs/2202.10446>. Physics-informed latent epidemic
  dynamics.
- **EpiLearn** (Liu et al., 2024) — <https://arxiv.org/abs/2406.06016> ·
  <https://github.com/Emory-Melody/EpiLearn>. The standardisation vehicle we
  should report through rather than a bespoke script.

### B4. Foundation models — do they beat GNNs?

- **Jafari, Fox, Fox, Marathe, Adiga (Jun 2026), "Understanding Key Features
  of Time Series Foundation Models from Epidemic Forecasting"** —
  <https://arxiv.org/html/2606.19560>. Chronos, TimeLLM, PatchTST,
  iTransformer, TimesNet, TiDE, TCN on 20 years of US ILI. **"a mixture-of-
  experts model that fuses multiple pretrained forecasters achieves the
  strongest overall performance."** LLM-style TimeLLM **"underperform[s]
  relative to numerical forecasters"**. No GNN comparison — the gap is mutual.
- **EpiCastBench**: TimesFM and Chronos-2 win outright over 13 models on 40
  datasets with significance testing.
- **QMUL medRxiv**: TabPFN-TS zero-shot beats every individual architecture.
- **Foundation models for time series forecasting and policy evaluation in
  infectious disease epidemics**, Epidemics 2026 —
  <https://www.sciencedirect.com/science/article/pii/S1755436526000320>.

**Answer:** on *univariate/multivariate* epidemic series, foundation models
now beat bespoke deep models and are competitive with mechanistic ones. They
do **not** yet exploit spatial structure. Nobody has run Chronos/TimesFM
against Cola-GNN/EpiGNN on Japan-Prefectures / US-Regions / US-States. **That
is a genuine, currently-unoccupied hole.**

---

## Part C — Synthesis

### C1. The table

| Paper | Year | Venue | Core idea | Gain over Cola-GNN / EpiGNN | Stated future work |
|---|---|---|---|---|---|
| CNNRNN-Res | 2018 | SIGIR | RNN + CNN cross-source fusion + residual links | (baseline) Japan h=3 1550 | — |
| LSTNet | 2018 | SIGIR | CNN + recurrent-skip + AR; graph-free | (baseline) Japan h=3 1459 | — |
| DCRNN | 2018 | ICLR | Diffusion conv in GRU enc-dec; traffic | (baseline, worst) Japan h=2 1502 | — |
| **Cola-GNN** | 2020 | CIKM | Loc-aware attention **gated with geo adjacency** + **multi-scale dilated conv** + message passing | Reference point | Iterative single model across leads; external indicators; identify drivers |
| STAN | 2021 | JAMIA | GAT+GRU with **SIR-constrained loss**; claims data | Not on this benchmark | Policy-aware dynamics; hospitalisation targets |
| MPNN+LSTM/TL | 2021 | AAAI | Mobility-graph MPNN + LSTM + **cross-country transfer** | Not on this benchmark | Transfer across outbreaks |
| CAMul | 2022 | WWW | Probabilistic **multi-view** with per-view uncertainty | Not on this benchmark | Calibration in hierarchies |
| **EpiGNN** | 2022 | ECML-PKDD | Degree-based **transmission risk encoding** + RAGL with **degree-gated geo adjacency** + multi-scale dilated conv + AR | **5.6%** vs best flu baseline; loses at Japan h=10 | "consider the time decay effects of spatial transmission" |
| MepoGNN | 2022 | ECML-PKDD | **Metapopulation SIR ⊕ GNN** | Japan COVID + mobility | Mobility generation |
| EINNs | 2023 | AAAI | Physics-informed latent epi dynamics | Not this benchmark | — |
| MSGNN | 2024 | DMKD | Multi-scale **graph** | US COVID only | — |
| BDSTGNN | 2024 | KBS | Backbone ⊕ dynamic graph; DLinear | 2 datasets | Information-theoretic graph analysis |
| **GNN survey** | 2024 | KDD | Taxonomy + agenda | — | Scales, cross-modality, **continuous** diffusion, interventions, explainability, noisy/private data |
| EARTH | 2024/25 | ICML | **Neural ODE** SIR + DTW-guided graph | Beats ColaGNN/EpiGNN under its own protocol | — |
| HeatGNN | 2024/25 | preprint | Mechanistic heterogeneity; EpiGNN backbone | **+1.6% (JP), +1.9% (USR), +4.1% (USS)** | Other epidemics; interpretable affinity graph |
| PISID | 2025 | PLOS ONE | STID embeddings **with no graph** + SIR | Competitive, graph-free | — |
| EASTG | 2025 | IEEE BIBM | Adaptive graph + next-generation matrix | 3 datasets | — |
| EpiHybridGNN | 2025 | arXiv | Cola-GNN ⊕ EpiGNN merge | Claims to beat both | — |
| **STTGNN** | 2026 | KBS | **Multi-scale temporal conv + ST Transformer + region-aware graph learner with degree-aware gating + GCN** | Claims SOTA on Japan + US-Regions | (paywalled) |
| **SpatialEpiBench** | 2026 | arXiv | 11 datasets, **rolling-origin**, outbreak metrics | **Cola-GNN, EpiGNN, EARTH mostly lose to naive** | Probabilistic metrics; rethink geographic adjacency |
| **EpiCastBench** | 2026 | arXiv | 40 datasets, Friedman/MCB | **TimesFM/Chronos-2 win**; GNNs untested | Spatiotemporal + probabilistic + mechanistic |
| M-SPICE | 2026 | KDD | Multimodal spatial maps ⊕ transformer | **ColaGNN 0.291 vs Persistence 0.213** | Multi-scale (HSA-level) |
| MultiFoundation | 2026 | arXiv | **MoE over pretrained forecasters** | Best overall; TimeLLM underperforms | UQ; mechanistic structure |

### C2. What the field says is open

**(i) Evaluation rigour — the loudest and most recent theme.** Chronological
single splits "do not reflect real-time forecasting practice"
(SpatialEpiBench); rolling-origin retraining is the standard being imposed.
Point-forecast RMSE/PCC is out of step with operational practice, where
**WIS, relative WIS, coverage and CRPS** rule. No significance testing in the
GNN literature; EpiCastBench introduces Friedman + MCB. **No naive/persistence
baseline** in Cola-GNN or EpiGNN — and when one is added, most of the family
loses. And **cross-paper numbers are not comparable**: Cola-GNN reports its
own US-States h=5 RMSE as 202, EpiGNN copies 202, EARTH re-runs it and gets
299.1. There is no canonical benchmark, only a shared filename.

**(ii) Mechanistic / epidemiological grounding.** Where every strong 2023–2026
paper went: STAN's SIR loss, MepoGNN's metapopulation SIR, EINNs, EARTH's
neural-ODE SIR, HeatGNN's mechanistic heterogeneity, EASTG's next-generation
matrix, CSTGNN, PISID. **A purely architectural model with no epidemiological
semantics is now the minority position and the one reviewers punish.**

**(iii) Uncertainty quantification.** The largest gap between the ML
literature and the deployed literature. SpatialEpiBench excludes probabilistic
metrics because "most methods produce point forecasts only" — an indictment,
not a limitation. Cola-GNN, EpiGNN, MSAGAT-Net: all point forecasts.

**(iv) Transfer / generalisation across outbreaks.** MPNN+TL (AAAI 2021) is
still the reference and it is five years old.

**(v) Data and mobility.** SpatialEpiBench's third failure mode: **"limited
utility of common geographic adjacency for epidemiological spatial
information"** — the prior this model biases toward. Survey §5.1: multi-scale
granularity is a *data/graph* problem, not a *convolution-dilation* problem —
**if the paper cites the survey's "multi-scale" as motivation for dilated
temporal convolutions, a reviewer will catch it.**

**(vi) Interpretability.** Survey §5.5: "neural models investigated thus far
have not placed significant emphasis on this aspect." Attention heatmaps
(Cola-GNN Fig. 3, EpiGNN Fig. 5) are now regarded as illustrative, not as
interpretability.

### C3. Is the MSAGAT-Net idea saturated? Yes.

| MSAGAT-Net component | Direct precedent | Where |
|---|---|---|
| Multi-scale **dilated** temporal conv | **Cola-GNN §3.3 "Multi-Scale Dilated Convolution"**, K=10, dilations 1/2 | CIKM 2020 |
| " | **EpiGNN §3.2 "Multi-Scale Convolutions"**, five filters | arXiv 2208.11517 |
| " | **STTGNN**, short/medium/long-term decomposition | KBS 2026 |
| Spatial attention with **learnable adjacency bias** | **Cola-GNN Eqs. 4–6**: `Â = M ⊙ Ã^g + (1−M) ⊙ A` | Cola-GNN §3.2 |
| " | **EpiGNN Eqs. 10–11**: `Ā = Dˢ ∘ A^geo + Â`, `Wˢ ∈ ℝ^{N×N}` learnable | EpiGNN §3.4 |
| " | **STTGNN**: "adaptively fused with static geographic information via a **degree-aware gating** mechanism" | KBS 2026 |
| " | **EARTH GLTG**: mask "to balance static geographic and learned dynamic connections" | ICML 2025 |
| **Progressive propagation** | **Cola-GNN §3.4 "Graph Message Passing – Propagation"** | Cola-GNN §3.4 |
| " | **EpiGNN §3.5** 1–5-layer GCN; **STTGNN** "lightweight GCN… over dynamically inferred graphs" | — |
| **All four jointly** | **STTGNN**, on Japan-Prefectures + US-Regions, RMSE/PCC/peak error | KBS, April 2026 |

Worse: Cola-GNN's ablation table already reports the marginal value of each
block. `w/o temp` at Japan h=2 is *better* than full Cola-GNN (912 vs 929).
Their own reading — "adding temporal and spatial modules does not change the
short-term prediction very much" — means the multi-scale temporal block was
known to be near-null at short horizons **in 2020**.

**So: yes, saturated.** Recombining attention, an adjacency prior, dilated
temporal convolutions and message passing on 47/49/10-node weekly series with
174–392 training points is not a 2026 contribution.

### C4. Numbers to beat — and the trap

**The trap.** There is no canonical protocol. Cola-GNN: 50/20/30, T=20,
h ∈ {2,3,5,10,15}, 10 seeds, RMSE/MAE/PCC. EpiGNN: same split but 5 seeds,
h ∈ {3,5,10,15}, RMSE/PCC only, and it **copies** Cola-GNN's numbers. EARTH
re-runs and gets Cola-GNN US-States h=5 = **299.1** where the original reports
**202**. HeatGNN re-runs and gets Cola-GNN Japan h=2 = **1168** where the
original reports **929**. **Therefore: any accuracy claim against published
numbers is uninterpretable unless every baseline is re-run under one declared
protocol.** Stating this explicitly is the cheapest credibility gain available
and turns a weakness into a methodological point.

**The canonical Cola-GNN-protocol ladder** (50/20/30, T=20; RMSE):

| Dataset | h | Cola-GNN | EpiGNN | The bar |
|---|---|---|---|---|
| Japan-Prefectures | 3 / 5 / 10 / 15 | 1051 / 1117 / **1372** / 1475 | **996 / 1031** / 1441 / **1470** | 996 / 1031 / 1372 / 1470 |
| US-Regions | 3 / 5 / 10 / 15 | 636 / 855 / 1134 / 1203 | **589 / 774 / 984 / 1061** | 589 / 774 / 984 / 1061 |
| US-States | 3 / 5 / 10 / 15 | 167 / 202 / 241 / **237** | **160 / 186 / 220** / — | 160 / 186 / 220 / 236 |

Australia-COVID (h=3/7/14, EpiGNN protocol): **EpiGNN 71.42 / 153.07 /
287.90**; Cola-GNN 127.59 / 279.56 / 326.79.

**2024–2026 claimants to cite and reconcile:** HeatGNN (h=2 RMSE): Japan
**1149**, US-Regions **541**, US-States **142**, Australia **315**. EARTH
(h=5/10/15): US-States **243.2 / 277.8 / 300.1**; Australia **156.8 / 177.6 /
225.3**. STTGNN: claims SOTA on Japan and US-Regions — **numbers not
obtainable, paywalled**. M-SPICE: ColaGNN 0.291 vs Persistence 0.213 vs
M-SPICE 0.194 NRMSE.

**And the baseline that will sink the paper if a reviewer asks: naive
persistence.** Absent from Cola-GNN and EpiGNN, and SpatialEpiBench found
that on 11 datasets "every method beats the naive baseline less than 50% of
the time". **Run it before resubmitting anywhere.**

---

## Gaps not closed

1. **STTGNN full text and results table** — ScienceDirect paywall (403).
   Architecturally the closest paper to MSAGAT-Net and the one that will be
   cited against it. Retrieve via the Coventry library.
2. **EpiHybridGNN numeric tables** — PDF stream not parseable.
