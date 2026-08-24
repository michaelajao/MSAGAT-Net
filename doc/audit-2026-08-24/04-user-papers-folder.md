# 04 — The papers in `doc/GNN forecasting/`, read in full

Eight PDFs supplied 24 Aug 2026, plus Fritz et al. (2022) fetched from
Nature, plus one ESWA 2025 paper supplied 25 Aug. Each was read cover to
cover. The question asked of each: *what does it do, what data does it need,
could it run on our benchmark, and what does it leave open?*

**Headline: none of them runs on counts + geographic adjacency as published.**
Their value is (a) Fritz independently reaching our failure-mode conclusion,
(b) a citable protocol hole in every one of them, and (c) one unoccupied
research axis that became Paper C.

---

## 1. Wang (2023), *Advances in spatiotemporal graph neural network prediction research*

Yi Wang, *International Journal of Digital Earth* 16(1):2034–2066.
DOI 10.1080/17538947.2023.2220610. Carries a published correction
(<http://dx.doi.org/10.1080/17538947.2023.2239591>) — cite the corrected
version.

### The taxonomy, and where MSAGAT-Net sits

Three orthogonal axes (§2.3, pp. 2038–2044):

- **Temporal**: RNN-based / CNN-based / Attention-based (+ MLP-based,
  GCN-based residual classes).
- **Spatial**: spectral-domain vs spatial-domain graph convolution. **GAT is
  filed under spatial-domain convolution, not as a third category.**
- **Graph type** (p. 2041, including the paper's own arithmetic error — "It is
  divided into three categories, namely static, dynamic, multi-scale and
  adaptive graphs"): static / dynamic / multi-scale / **adaptive**, the last
  defined as "the dynamic graph that does not require predefined adjacency
  matrix, which is trained directly by learnable node embeddings… The
  adaptive adjacency matrix typically consists of two node embedding vectors
  that undergo matrix multiplication."

That definition describes MSAGAT-Net's low-rank `U@V` verbatim.

**MSAGAT-Net's Table 2 row would read:**
`CNN-Based, Spatial domain graph convolution | Dilated/Separable Convolution,
Attention Mechanism, Gated Mechanism, Graph Generation | Adaptive`

**The occupants of that exact cell:**

- **STGAT (Kong et al. 2020)** — *the closest row*: "CNN-Based, Spatial domain
  graph convolution | GAT, Dilated Convolution, Gated Mechanism, CNN, Graph
  Generation | **Adaptive**". Attention + dilated temporal conv + gating +
  learned graph.
- **Graph WaveNet (2019)** — diffusion multi-hop + dilated TCN + adaptive
  adjacency.
- **MTGNN (2020)**, **AGCRN (2020)**, **FC-GAGA (2021)**, **HAGEN (2022)**,
  **AST-InceptionNet (2023)**, **Tres2GCN (2022)**.

**Negative finding worth citing:** there is **no Graphormer, no
structural/additive attention bias, and no graph-transformer literature
anywhere in this survey**. Every attention model here *re-weights*; none
*biases*. That is a gap in *this survey*, not in the field (Graphormer is
2021) — claim only "under-explored in the ST-GNN forecasting lineage surveyed
by Wang (2023)".

### Epidemic coverage: 2 models out of 59

Only **CovidGNN** (Kapoor et al. 2020, "MLP-Based, Spectral | GCN, MLP |
Static") and **STAN** (Gao et al. 2021, "Attention-Based, Spatial | GAT, GRU
| Dynamic"). **Neither uses an adaptive adjacency. Neither uses multi-hop
diffusion.** No epidemic accuracy numbers anywhere; no epidemic entry in the
timing or comparison tables.

The one epidemic-relevant methodological statement, §2.4.2 (p. 2045):

> "Sparse spatiotemporal graph data prediction mainly includes… crime case
> prediction, infectious disease prediction… their data values are zero in
> most cases… This type of model must be aware of the problems arising from
> zero-value inflation… The current situation is mostly improved by using
> reinforced loss functions, or more sensitive model components."

**This is the single best sentence in the paper for our framing** — it names
zero-inflation as the open problem for sparse epidemic data, which is exactly
what Paper C's NB/ZINB likelihood answers.

### Its stated open problems (§3.2, pp. 2057–2058, verbatim openings)

1. **Deepen network and broaden feature** — "the network can only be stacked
   in layers 1–3, and the accuracy slowly decreases from layer 4 onwards."
2. **Multivariate forecasting** — "Most of the current networks only consider
   the prediction data itself, and ignore the influence of some other
   factors… the difficulty of this challenge lies in the unavailability of
   data."
3. **Self-adaptive adjacency matrix** — "In models such as Graph WaveNet,
   STGAT, AGCRN, AST-InceptionNet, adaptive adjacency matrices are utilized…
   The experimental results found that the adaptive adjacency matrix can
   **completely** learn the previous graph structure's adjacency relations…
   **For the vast majority of current networks, this is not yet possible.**"
4. **Multi-scale spatiotemporal neural networks** — graph pooling / clustering
   for local subgraph structure.
5. **Temporal pattern mining** — "Most of the networks merely focus on
   proximity, yet ignore the trend and periodicity."

**Read item 3 carefully.** It is not "this is an unsolved research problem";
it is "**this is solved and under-adopted**" — a diffusion gap, not a
knowledge gap. Any reviewer reading §3.2.3 will read it the same way.

### Evaluation practice — a free target

The survey compares 59 models by **copying numbers out of their own papers**
(p. 2047: "These results are cited from literature"), and its fairness
criterion is (p. 2049): "If their evaluation results of benchmarks are the
same or less different (**no more than 5%**) from the original paper, it
indicates that their proposed model defaults to a fair comparison." It
**never mentions** significance testing, confidence intervals, repeated runs,
seed variance, error bars, hyperparameter budgets, or split protocols.

Interpretability is treated as self-evident ("expresses the strength of
information in the form of scores"), with **no discussion of
attention-as-explanation faithfulness and no validation of learned adjacency
against ground truth**.

### Verdict

"Adaptive adjacency + attention bias + multi-hop propagation" is a **solved
cell** in this survey's own framing, established on **dense traffic data**.
The transferable observation is that the mature dense-data toolkit has, by
this survey's accounting, **never been applied to sparse epidemic data** —
which reframes MSAGAT-Net honestly as a transfer of that toolkit, not a new
one.

---

## 2. Kosma et al. (2023), GN-ODE — *Neural ODEs for Modeling Epidemic Spreading*

Kosma, Nikolentzos, Panagopoulos, Steyaert, Vazirgiannis, *TMLR* 08/2023,
18 pp. OpenReview `yrkJGneOuN`.

**Mechanism.** The ODE *is* the individual-based SIR mean-field system (p. 4,
Eq. 1):

```
dS/dt = −β(A I_h) ⊙ S_h
dI/dt =  β(A I_h) ⊙ S_h − γ I_h
dR/dt =  γ I_h
```

with `A` the **fixed** contact-network adjacency. β and γ are scalars, **not
learned** — given per SIR instance, with uniform rates assumed (p. 3). The
network only encodes binary node indicators, refines per solver step, and
decodes a softmax 3-vector. Euler solver, step 0.5, adjoint gradients. The
authors note `A I` "can be seen as a form of message passing" — the SIR term
*is* an MPNN aggregation.

Ablation ODE-RK (no trainable parameters at all): MAE 0.09608 / 0.10653 /
0.19109 vs GN-ODE 0.05631 / 0.01527 / 0.01924 — **the neural refinement does
the heavy lifting, not the SIR prior**.

**Data required.** Individual-level contact networks (nodes = people);
per-node binary initial conditions; **known β and γ**; and **simulated
ground truth** ("10⁴ simulations for 20 time-steps").

**Can it run on our benchmark? No — and it is not a forecasting paper.** It
is a *simulation-surrogate* paper: no observed time series, no lookback
window, no autoregressive rollout, no noise. Targets are Monte-Carlo marginal
probabilities in [0,1] summing to 1 — structurally incompatible with
unbounded weekly ILI counts.

**Protocol.** 8 social networks (karate 34 → Epinions 75,877); 200 SIR
instances each; β,γ ~ U[0.1,0.5]; **60:20:20**; **5 repeats, mean ± std**.
Best on 5 of 8; DMP wins the three largest graphs.

**Two protocols worth stealing:** (i) **OOD generalisation** — bin β,γ into 5
bins, train on bins 2–4, test on 1 and 5. (ii) **Cross-graph transfer** —
train on small graphs, test on unseen graphs 40× larger; GN-ODE's
train→unseen gap is small where GIN's is dramatic.

**Interpretability.** Claimed ("paving the way for the extensive application
of interpretable neural networks"), **validated by one heatmap** (Fig. 7).
No parameter-recovery experiment, no test that the refinement respects SIR
invariants.

**Limitations/future work: there is no such section.** Scattered admissions:
performance degrades on the largest networks; inference is slower than plain
GNNs because of the intra-solver step; "The accuracy of the above system
depends on how much the independence assumption… holds in practice."

**Cites:** MPNN (Gilmer 2017) twice. **Not Cola-GNN, not EpiGNN, not STAN.**
Cites Panagopoulos MPNN+TL (a co-author's own epidemic paper) but **never
compares against it**.

---

## 3. Shi, Zhang & Morris (2022), PAN-cODE — *COVID-19 forecasting using conditional latent ODEs*

*JAMIA* 29(12):2089–2095. doi:10.1093/jamia/ocac160. Brief Communication.
Code: <https://github.com/morrislab/PAN-cODE>

**Mechanism.** A conditional Latent ODE. Encoder `GRUODE` → `z0 ~ N(μ,σ²)`;
**the contribution** is splicing an NPI stringency vector `I_fc` into the
latent initial state, `z̃0 = [z0, I_fc]`, so the ELBO is conditioned on
policy. Decoder is **autoregressive**: `(i_t, d_t) = Linear(i_{t−1}, d_{t−1},
z_t)`, which "serves to restrict the maximum change in caseload between
timepoints". Reconstruction runs the ODE **backwards** from the forecast
date; forecasting runs forward.

**How the graph enters: it does not. There is no graph, no adjacency, no
spatial coupling.** Regions are independent trajectories sharing weights.

**What is fixed by epidemiology: nothing.** "the actual dynamical function is
learned directly from data, avoiding the need to manually specify an ODE
function as in traditional compartmental models like the SIR model" (p. 2093).

**Data required.** Daily cases and deaths (US state + county, ~2,500
trajectories); **Oxford OxCGRT NPI indices** (Stringency, Government
Response, Containment Health, Economic Support) as both conditioning and
covariates; daily temperature; 14-day lag on NPI features for deaths.
Explicitly: "PAN-cODE does not require mobility or hospitalization data."

**Can it run on our benchmark? Mechanically yes, substantively no.** With no
graph it ingests counts alone — but its entire contribution evaporates:
OxCGRT does not exist for the ILI datasets (2002–2019), and for the COVID-era
sets it is **national-level only**, so `I_fc` would be constant across all 372
LTLAs. Strip the conditioning and you get their **GRU-ODE baseline, which
performs badly** (MAE 371 vs 167).

**Protocol.** Cumulative deaths across 51 US regions; **two forecast dates**
(28 Dec 2020, 8 Mar 2021); **4 and 6 weeks ahead**; median absolute error and
mean rank; **Wilcoxon signed-rank / ranked-sum at p = .05**. **No seed count
is reported anywhere.** ~20 COVID-19 Forecast Hub models as baselines.

Results: **PAN-cODE does not win at 4 weeks** (167 vs Google_Harvard-CPF 118
at Dec-2020; 93 vs Covid19Sim 67 at Mar-2021) but **wins at 6 weeks** (207 vs
312; 80 vs 109). Their own summary: "it performs significantly better than
all existing methods on 6-week death forecasting from March 8, 2021, and never
performs significantly worse than the best performing method in all other
forecasting evaluation categories." **Parity at 4 weeks, win at 6 weeks** —
honest framing worth imitating.

Unseen countries (% error, 4wk): PAN-cODE −10.4 / 8.2 / 7.2 / 0.4 for Canada
/ UK / India / Russia vs PrevWeek 39.0 / 66.7 / 63.1 / 53.4. At 6 weeks it
under-forecasts by ~50% and still calls them "reasonable projections".

**Interpretability.** LIME feature importance (validated only by
agreement-with-intuition, with their own caveat that other methods "might
find different relationships"), and counterfactual trajectories from 100
sampled latents — **validated by visual plausibility only**.

**Limitations, verbatim:**

> "However, PAN-cODE does not explicitly learn a causal model between NPI
> stringency and future caseload. Building a formal causal model is likely
> difficult due to delayed and noisy reporting, and we leave this as future
> work."

And the single most important sentence for us (p. 2093):

> "However, it would be straightforward to incorporate a SIR compartmental
> model into PAN-cODE. By using the Neural ODE, PAN-cODE would be able to fit
> the dynamical parameters of this SIR model using backpropagation…"

**That is EARTH's core idea, written down as an unexecuted TODO in 2022.
That slot is now closed.**

Also asserted and **never tested**: "PAN-cODE is also capable of natively
handling datasets for pandemics where observations are sparsely or irregularly
observed."

**Cites STAN** (ref 20) and Kapoor et al., but **not Cola-GNN, not EpiGNN**,
and compares against neither.

---

## 4. Jin et al. (2023), MTGODE — *Multivariate Time Series Forecasting With Dynamic Graph Neural ODEs*

*IEEE TKDE* 35(9):9168–9180. doi:10.1109/TKDE.2022.3221989.
Code: <https://github.com/GRAND-Lab/MTGODE>

**Mechanism.** Two coupled ODEs.

*Continuous Graph Propagation* — Proposition 1 (Eq. 6):
`dH^G(t)/dt = (Â − I_N) H^G(t)`, with an attentive readout over intermediate
solver states. The theoretical point: the discrete form rigidly ties depth K
to integration time by forcing Δt = 1; decoupling `K = T_cgp/Δt_cgp` is what
kills over-smoothing.

*Continuous Temporal Aggregation* — Proposition 2 (Eq. 11):
`dH^T(t)/dt = P(TCN(H^T(t), t, Θ), R)` with gated dilated convolutions and
**multiple kernel widths m ∈ {2,3,6,7}**, chosen because "most of the time
series data have inherent periods (e.g., 7, 14, 24, 28, and 30)". **One
parameter set replaces L stacked layers.**

Graph structure learning is MTGNN's, **with no prior**:
`A_ij = ReLU(tanh(β(M¹_i M²ᵀ_j − M²_j M¹ᵀ_i)))` from randomly initialised
embeddings, sparsified, uni-directional.

**Epidemiology fixed: none. Zero.** A pure architecture paper.

**Data required.** Only the series. No adjacency (learned), no covariates.
The most portable of the three ODE papers in principle.

**Can it run on our benchmark? Format yes, statistics no.** Their datasets
are 17,544–52,560 timesteps; Japan-Prefectures is **348**. Their single-step
setting uses **input length 168** — half of a 348-step series. The Eq. 8
graph learner trains `2Nd + 2d²` free parameters with no supervision; on 47
nodes × 348 weekly steps it will overfit, and their own ablation shows the
learner is worth only 0.0020 RSE even at 26k timesteps. **And there is no
mechanism to use the geographic adjacency we already have** — Eq. 6 accepts
any Â, so injecting it is a one-line change, but that is *our* contribution,
not theirs.

**Protocol.** Electricity / Solar / Traffic / Metr-La / Pems-Bay; input 168;
**60/20/20**; horizons 3, 6, 12; RSE and CORR; "All experiments are
independently repeated ten times… Averaged performances are reported" — **10
runs, but no standard deviations in any table.**

**Margins over MTGNN are 1–4% RSE.** On Electricity h=3, MTGNN **beats**
MTGODE on CORR. On Pems-Bay 60-min, GMAN wins. The efficiency argument is
stronger than the accuracy argument (relative time/epoch: MTGODE ×1.0, MTGNN
×1.34, GMAN ×28.08) — and the headline efficiency figure is obtained with
deliberately degraded solver precision (p. 9179).

**Interpretability: none claimed.** And **the learned adjacency A is never
visualised, never compared to the known road/sensor topology on Metr-La and
Pems-Bay** — a free validation the authors skipped.

**Limitations/future work: no such section.** Inline concession: "we can find
a sweet spot when selecting the spatial or temporal integration time" — i.e.
dataset-specific tuning with no principled selection.

**Cites none of Cola-GNN, EpiGNN, STAN, MPNN. Zero epidemiology in 13 pages.**

---

## 5. Panagopoulos, Nikolentzos & Vazirgiannis (2021), MPNN+TL

*AAAI-21*. arXiv:2009.08388v5.

**Mechanism — there is NO epidemiological structure.** This is usually
misremembered. The model is a plain MPNN over a *mobility* graph; the
"epidemiology" is metaphorical. The entire mechanistic claim is the product
`A^(t) X^(t)`, read as "an estimate of the number of new latent cases in u…
broken down to the cases received from other regions and the new cases caused
due to mobility inside u". No compartments, no β, no γ.

**MPNN+LSTM**: MPNN per snapshot → two LSTM layers → output.

**Damning detail (p. 7), verbatim:** "since we rely solely on the number of
confirmed cases, we can not utilize models that work with recovery, deaths and
policies, such as SEIR… In some preliminary experiments, however, this
provided errors in a different scale then the ones mentioned here, similar to
Gao et al. (2020), which is why we have not experimented further."

**Data required.** Facebook Data for Good movement maps at NUTS-3 level.
Italy 105 regions, England 129, Spain 35, France 81. They state: "our open
data lacks in many cases the number of recovered cases, deaths and population
demographics required for training these models."

**Can it run on our benchmark?** **The model, no** — the graph *is* the
mobility matrix; replace it with binary adjacency and MPNN degenerates to a
vanilla GCN. **But the transfer protocol is fully runnable and
modality-agnostic.** This is the single most transportable thing in all four
hybrid/transfer papers.

**Protocol.** **Expanding-origin**: T starts at 14 days and grows one day at
a time, **a different model per T and per horizon**. Validation = days T−1,
T−3, T−5, T−7, T−9. **No seed averaging anywhere — single run per
configuration.** Baselines include **LAST_DAY (persistence)** and **TL_BASE**
(an MPNN trained on the other three countries pooled).

**Results (Table 2, avg error per region, 14-day column):**

| Country | MPNN (no TL) | **MPNN+TL** | Δ | TL_BASE (naive pooling) |
|---|---|---|---|---|
| England | 8.13 | **6.84** | −15.9% | 13.48 |
| France | 6.93 | **6.13** | −11.5% | 12.24 |
| Italy | 17.88 | **16.69** | −6.7% | 24.89 |
| Spain | 44.25 | **34.65** | −21.7% | 59.68 |

MPNN+TL wins 12/12 cells, but at 3 days the gain over AVG_WINDOW is only
**3–8%**; LSTM/ARIMA/PROPHET are all *worse than persistence* in most cells.
The authors concede: "even though the MPNN+TL outperforms the baselines, their
predictions are not very accurate in terms of average error."

**The decisive result is TL_BASE**: naive pooled pretraining is *worse than
training on the target alone* in all 12 cells. **So the gain is attributable
to MAML specifically, not to "more data".**

**Transfer, exactly.** First-order MAML, **leave-one-country-out**, **nothing
is frozen**. A task = (country k, training-set size i days, horizon j). Two
gradient steps (Eqs. 2–3), Hessian term dropped (Eq. 4) → FOMAML. The base
learner is the plain MPNN, not the LSTM variant. Then full fine-tuning of
every parameter on the target.

**Future work, verbatim:** "**Our final goal is to evaluate the model on the
second wave of COVID-19, based on the first.**" — never done; explicitly open.

---

## 6. Gao et al. (2021), STAN

*JAMIA* 28(4):733–743. Also summarised in
[03-literature-baselines-and-followons.md](03-literature-baselines-and-followons.md) §A4.

**Mechanism.** SIR enters as a **training-time-only auxiliary loss**, never as
a forward simulator. Graph edges are a **fixed gravity model, not learned**:
`w_ij ∝ p_i^α p_j^β exp(−d_ij/r)`. Two-layer multi-head GAT → MaxPool across
nodes → GRU → two heads: `β,γ = sigmoid(MLP(h))` and `ΔÎ, ΔR̂ = MLP(h)`.

**The key line (p. 737), verbatim:** "Note that we do not use the transmission
dynamics constraints in the prediction time because this module is only used
for optimizing the model in training time." **The deployed forecast is the
free MLP head. SIR is a regulariser, full stop.**

**Data required.** **Recovered counts R** (both loss terms need ΔR),
**population N_P**, lat/lon/density for the gravity graph, and the **IQVIA
US9 claims database** (453,079 patients, 48 COVID-19 ICD-10 codes) as dynamic
node features.

**Can it run on our benchmark? No, not as published.** And it is
**conceptually invalid** on multi-season ILI: those are consultation rates,
not infections; there is no observable susceptible pool and no monotone
depletion over 348 weeks for `βS` to describe. Note Panagopoulos
independently reports that a simple mechanistic model on counts-only data gave
"errors in a different scale".

**Protocol.** Mar 22 – Jun 10 2020; test starts May 17. **Single temporal
split. No rolling origin. No seed averaging.** 95% CIs from bootstrap over
data, not model randomness. **There is no naive/persistence baseline anywhere
in the paper** — and the test window is a **monotone growth phase** (their
Figs. 3–4 show all four example curves rising), precisely where persistence is
strong.

Results vs ColaGNN (state, L_P=15): STAN MSE 972,192 vs 7,192,031 — the
abstract's "87%" is this single best cell. Ablations: the SIR constraint
roughly halves MSE; the graph is worth ~3×.

**Interpretability — claimed but never validated.** STAN learns
location-and-window-specific β,γ, and **never reports, plots, or checks a
single value**. No time series of β, no comparison with published R₀. The only
evidence the constraint "means something" is that the ablation hurts accuracy.
**This is an open, citable gap.**

**Limitations, verbatim (p. 738):** three items — the prediction-window
setting ("if the number of cases fluctuates drastically… it is difficult for
the STAN model to learn valid stable transmission and recovery rates"); "the
transmission dynamics constraints may be too simple"; and data quality for the
attribute graph.

---

## 7. Nikparvar et al. (2021), MTS-LSTM — *Spatio-temporal prediction of COVID-19 in US counties*

*Scientific Reports* 11:21715. doi:10.1038/s41598-021-01119-3.

**Mechanism — no GNN, no graph, no epidemiological structure.** Their own
admission (p. 10): "**we did not account for spatial dependencies between
counties in this research.** All counties that participate in predictions of
dynamics for a county have the same weights."

A two-layer LSTM (128,128) → Dense(32) → dropout → Dense(3), predicting new
cases, deaths and foot traffic simultaneously, iterated 4 weeks ahead. The one
structural idea: pool feature vectors from **all** counties into a shuffled
bag and train **one global model** — now the default in every modern
benchmark, so the contribution has aged into a non-contribution.

**Data required.** JHU CSSE + **SafeGraph foot traffic** (~18M devices). They
screened a wide covariate set and concluded: "COVID-19 is predicted with the
highest accuracy with the SafeGraph mobility data as the sole covariate
feature" and "our analysis did not support using a series of covariates as
predictors."

**Can it run on our benchmark? Yes trivially** (Model 1 uses cases + deaths
only) — but then the paper's empirical claim is untestable and what remains is
"train one pooled LSTM", which is the baseline, not a method.

**Protocol.** 33 weeks, ~3100 counties, **last 4 weeks held out, single
split**; 70/30 train/val repeated ten times with **randomly selected counties**
(not seeds). **Baseline: exactly one** — the CDC COVID-19 Forecast Hub
ensemble. No persistence, no ARIMA.

**Result: they lose to the ensemble.** RMSE_total new cases **224.28**
(Model 1) / **169.84** (Model 2) vs **159.92** (Ensemble). And mobility makes
*death* prediction worse (3.44 → 4.09). Their own framing: "**While this may
not add a new capability in modeling performance**, our MTS-LSTM model has
several attractive advantages" — the sell is parsimony, not accuracy.

**Future work, verbatim, two items directly relevant to us:**

> "**The integration of LSTM with convolutional layers in graph neural
> networks is also an interesting line of future research for disease spread
> prediction.**"
> "**It will be important to use this pre-trained model to predict disease
> dynamics in other geographic contexts, time periods, or even similar
> infectious diseases such as influenza in order to demonstrate the
> reproducibility of our methodology and results.**"

The second is a *second, independent* call for cross-outbreak transfer, never
executed.

---

## 8. Fritz, Dorigatti & Rügamer (2022) — *Combining GNNs and spatio-temporal disease models*

*Scientific Reports* 12:3930. doi:10.1038/s41598-022-07757-5. Preprint
arXiv:2101.00661. **The most methodologically careful of the set, and the one
whose machinery survives without mobility.**

**Mechanism — semi-structured deep distributional regression + GNN, one joint
likelihood, orthogonalised. There is no SIR.** The "epidemiological structure"
is an endemic–epidemic regression (Held/Meyer `hhh4` lineage) on lagged
incidence, inside a count likelihood.

Zero-inflated count distribution (Eq. 3):
`f_D(y|λ,π,χ) = π I(y=0) + (1−π) f_C(y|λ,χ)`, reparameterised so one additive
predictor drives both.

The fusion (p. 10): `η_k,ig = x_ig(t) ϑ_k^str + ũ_k ϑ_k^unstr`.

**The actual innovation — orthogonalisation (p. 10), verbatim:**

> "Identifiability is crucial to our analysis since some feature information
> is shared between the structured effects and unstructured effects… If these
> two model parts are not adequately disentangled, it is unclear what part of
> the model is accounting for which information in the shared features.
> Therefore, the latent GNN representations u_i are orthogonalized with
> respect to z_ig(t)… we project u_i in the orthogonal complement of the
> column space spanned by z_ig(t)."

Plus a `log(pop)` offset so the target is a rate, a MoNet-style GNN over a
**fully connected** 401-district graph, and a single joint penalised
likelihood (RMSprop).

**A sharp reading almost nobody states: Fritz's GNN sees no time.** Table 1
assigns to the GNN only **static** node attributes (population, density) and
edge attributes (social connectedness, distance, adjacency). Verbatim (p. 10):
"The unstructured part of our network computes each district's embedding
(node) by exploiting **time-constant** district population attributes and edge
attributes." The time-varying mobility enters through the **structured
splines**, not the GNN. So the GNN contributes a *learned static spatial
embedding* — effectively a spatial random effect. This explains why standalone
GNN collapses on later folds (5.972 → 49.064 → 77.162).

**Data required.** Facebook colocation → Gini index; Social Connectedness
Index (an April-2020 time-constant snapshot) → MDS embeddings; staying-put
percentage; RKI cases by disease-onset date stratified by age and gender, with
~30% missing onset dates **imputed via a learned probabilistic delay model**;
population and density.

**Can it run on our benchmark? Mostly yes — the most transportable of the
four.** Strip Facebook and you retain: ZIP/NB likelihood head + penalised
spline on lagged incidence + `log(pop)` offset + GNN over adjacency +
orthogonalisation + the epistemic/aleatoric machinery. Three caveats:
(i) zero-inflation only bites where counts are low (LTLA, Australia);
(ii) the demonstrated interpretability payoff is entirely about mobility;
(iii) **the paper reports no ablation removing mobility**, so its own central
claim ("the necessity of including mobility data") is unquantified.

**Protocol.** Expanding window, **6 folds** (test weeks 32/35/38/41/44/47 of
2020), **horizon = 1 week only**, RMSE. Baselines include **MEAN** (a genuine
persistence-family baseline), GAM, XGBoost, DNN, and **GNN alone**.

**Table 2 — RMSE:**

| Model | wk 32 | wk 35 | wk 38 | wk 41 | wk 44 | wk 47 |
|---|---|---|---|---|---|---|
| XGBoost | 4.926 | 5.188 | 7.447 | **15.327** | 65.036 | 74.235 |
| DNN | 10.179 | 12.178 | 17.897 | 64.065 | 108.474 | 80.901 |
| GAM | 4.042 | 4.738 | 4.736 | 21.666 | 18.556 | 23.813 |
| MEAN | 5.038 | **3.666** | 6.196 | 30.910 | 20.090 | 23.159 |
| GNN alone | 5.972 | 6.785 | 11.355 | 49.064 | 77.162 | 53.489 |
| **Ours (ZIP)** | **3.931** | 4.235 | **4.500** | 16.588 | **17.738** | **15.050** |
| Ours (NB) | 4.096 | 4.094 | 5.174 | 28.580 | 18.098 | 31.724 |

**At week 35 the naive MEAN (3.666) beats every other model.** Margin over
GAM: +3% / −11% / +5% / +23% / +4% / +37% — **the fusion buys almost nothing
in calm periods and 23–37% at inflection points.**

**The sentence that matters most for Paper B**, from their conclusion:

> "The given findings also highlight the need for regularization and showcase
> how common ML approaches can **not adequately capture the autoregressive
> term, which, in turn, proved to be essential for the forecast**."

**This is our finding — EAGAM collapses to mean pooling while the highway/AR
and damped-trend blend carries the model — reached independently on German
district data.**

**Interpretability — the strongest of the set, and actually validated.**
Bivariate partial effects of (week, Gini) and (week, staying-put) with
narrow CIs; partial effect of lagged infection rate; **epistemic uncertainty
validated against error (Spearman ρ = 0.76 between 10-network ensemble SD and
absolute error, "grows approximately linearly with the error")**; calibration
checked with one-sided t-tests. Honest self-report on aleatoric intervals:
they "cover on average over 80% of all cases" — i.e. **under-coverage** for a
nominally ~95% interval.

**Future work, verbatim:** "First, using time-varying, as opposed to static
networks… Second, **the semi-structured approach of this article could be
extended to incorporate epidemiological models such as SIR as a third additive
predictor.** Finally, additional data sources… e.g., daily instead of weekly
infection count."

---

## 9. Yin et al. (2025), STGNN — *business process performance* (ESWA)

Yin, Qiu, Fang, Wang, Dong, Ge, *Expert Systems With Applications* 291:128391.
doi:10.1016/j.eswa.2025.128391.

**This is not an epidemic-forecasting paper.** It is business-process mining
on shipbuilding event logs (N = 14 sub-processes, one feature, 24-h
aggregation, proprietary data). It cites **none** of Cola-GNN, EpiGNN, STAN,
MepoGNN, MPNN, DCRNN, Graph WaveNet, MTGNN, AGCRN, Graphormer, PDFormer,
EARTH, HeatGNN or STTGNN. Its only forecasting ancestor is ASTGCN (2019),
whose temporal/spatial attention equations it reuses.

**Its value to us is as a calibration sample for ESWA, Paper B's fallback
venue.** What ESWA accepted: a reframed problem (individual-instance →
network-level prediction) plus a competent assembly of off-the-shelf blocks
(ASTGCN attention + GAT + MLP), evaluated with **one proprietary dataset, one
metric (MAE), four baselines of which the only non-deep one is a historical
average, one ablation, a single 80:20 split, no validation set, no seeds, no
std, no significance tests, no parameter counts, and no code release.**
Hyperparameter sweeps (Fig. 10) had no untainted set to run on.

**Implications:** (1) a new operator is **above** ESWA's bar, not at it — a
well-motivated domain reframing suffices; (2) Paper B's evaluation would
exceed this bar substantially and visibly; (3) ESWA reviewers will want a
domain-grounded story for each block, phrased in application terms, because
every equation in this paper is justified by a managerial narrative.

Its ablation is worth one number: removing the GAT branch costs
**4.66 / 4.70 / 8.39 / 7.21%** at 1/3/5/7 days — **the explicit adjacency
branch is the most important component and its value grows with horizon**,
consistent with our own horizon-threshold finding (E11).

---

## Synthesis

### What the hybrid / continuous-time line shares

1. **Nobody simulates.** The structure is always a soft constraint or an
   additive term. STAN's SIR is discarded at inference; Fritz has no ODE at
   all. "We embedded epidemiological dynamics into a deep model" oversells
   what is in the code in both cases.
2. **The mechanistic prior is not the load-bearing element; the
   autoregressive term is.** Fritz says it outright; his GNN alone scores
   49–77 RMSE where the full model scores 16–18. STAN's no-GNN ablation still
   beats every published baseline.
3. **Gains concentrate at regime changes and long horizons, not on average.**
   STAN 59% → 87% as L_P goes 5 → 15; Fritz 3–5% in calm weeks and 23–37%
   during the second wave; Panagopoulos 3–8% at 3 days and 12–22% at 14 days;
   Yin 4.7% → 8.4% as horizon grows. **This is the one reproducible finding
   across all of them, and it matches our E11.**
4. **Baseline hygiene is systematically poor.** STAN: no persistence baseline
   at all, on a monotone-growth test window. Nikparvar: no naive baseline, and
   loses to the ensemble it benchmarks against. Panagopoulos: LSTM/ARIMA/
   PROPHET all lose to `LAST_DAY`. Fritz is the only one who includes a
   rolling mean — and it beats his model in a fold. **Every headline
   percentage in this literature is inflated by baseline selection.**
5. **None of the four reports seed-averaged point forecasts.**

### What our benchmark lacks

| Paper | Needs beyond counts + adjacency | Verdict |
|---|---|---|
| Panagopoulos | Facebook movement maps (the graph *is* mobility) | Model **not runnable**; **transfer protocol fully runnable** |
| STAN | Recovered counts, population, lat/lon, IQVIA claims | **Not runnable**; SIR conceptually invalid on multi-season ILI |
| Nikparvar | SafeGraph foot traffic | Runnable, but then it is a pooled LSTM |
| Fritz | Facebook colocation/SCI/staying-put; age-gender strata | **Mostly runnable** — likelihood head, AR spline, offset, GNN-on-adjacency, orthogonalisation, uncertainty all survive |
| Kosma | Contact networks, known β/γ, MC ground truth | **Not runnable**; not a forecaster |
| PAN-cODE | OxCGRT NPI indices | Runnable but contribution evaporates; no graph |
| MTGODE | Nothing extra | Format-compatible, statistics-incompatible (348 vs 26k timesteps) |

### What is closed, and what is open

| Open item, in the authors' words | Status |
|---|---|
| Fritz: "extended to incorporate epidemiological models such as SIR as a third additive predictor" | **Closed** — STAN, MepoGNN, EINNs, EARTH, HeatGNN, PISID, CSTGNN |
| PAN-cODE: "it would be straightforward to incorporate a SIR compartmental model" | **Closed** — this is EARTH |
| Fritz: "time-varying, as opposed to static networks" | **Mostly closed** by dynamic-graph work |
| STAN: learns β,γ but never reports or validates them | **OPEN.** No paper in this set validates learned epidemiological parameters against anything external |
| Fritz: orthogonalisation as an identifiability constraint | **OPEN.** Essentially unused in the epidemic-GNN benchmark line |
| Panagopoulos: "evaluate the model on the second wave, based on the first" | **OPEN, never executed** |
| Nikparvar: "predict dynamics in other geographic contexts, time periods, or even similar infectious diseases such as influenza" | **OPEN, never executed** |
| PAN-cODE: native handling of sparse/irregular observation | **OPEN** — asserted twice, tested by nobody |
| MTGODE: principled selection of integration time | OPEN, minor |

### The unoccupied axis (this became Paper C)

**Do not build:** another SIR-embedded GNN (seven systems occupy it; and on
ILI-rate benchmarks the SIR state is unobservable); a neural-ODE forecaster
with a free-form learned adjacency (MTGODE, on far more data, with theory);
node-level SIR-probability prediction on contact networks (GN-ODE).

**Do build, in descending order of defensibility:**

1. **Identifiability-constrained likelihood decomposition** (Fritz's real
   innovation, minus mobility): NB/ZINB head + `log(pop)` offset; a penalised
   structured autoregressive/seasonal component; a **time-varying** GNN
   residual **orthogonalised** against it; one joint likelihood. Delivers
   (i) calibrated intervals with an epistemic/aleatoric split validated the
   way Fritz validated his, and (ii) **an identified answer to "how much of
   this forecast is plain autoregression versus genuine spatial spillover"** —
   the question the whole spatiotemporal-GNN literature dodges by reporting
   only aggregate MAE. Fritz's own GNN is *time-constant*, so an ordinary
   time-varying version is already a strict improvement on his design.
2. **Cross-dataset / cross-season FOMAML transfer** with counts + adjacency as
   the only shared modality — *strengthened* by our benchmark being
   modality-poor, because the only thing that **can** transfer is the dynamics
   operator. `TL_BASE` makes it non-trivial. Both Panagopoulos and Nikparvar
   name it as unexecuted future work.
3. **Endogenous conditioning on R̂** (PAN-cODE's `z̃0 = [z0, I_fc]` with the
   NPI index replaced by an EpiEstim or renewal-decoder R̂ derived from counts
   alone) — sidesteps exactly the limitation PAN-cODE flags.
4. **Robustness to irregular/missing observations** — asserted by PAN-cODE,
   tested by nobody, and impossible for fixed-window discrete models by
   construction.

**Honest expectation from this literature:** on counts-only data with
geographic adjacency, the achievable margin over a well-tuned
persistence/moving-average baseline at short horizons is **3–10%**. The
defensible contributions live at long horizons, at regime changes, in
calibration, and in transfer to data-scarce onsets — not in average-case point
accuracy.
