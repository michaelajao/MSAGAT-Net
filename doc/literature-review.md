# Literature review: spatiotemporal epidemic forecasting, 2018–2026

Close reading of 35 papers and four codebases (Cola-GNN, EpiGNN, BDSTGNN,
TERN), done 29 September 2026 for the one-paper plan in ledger §0. Six
readers each took a group of papers and recorded, per paper: claims (quoted),
method, the exact evaluation protocol, whether baseline numbers were re-run or
copied, headline numbers, ablations, how the paper is written, red flags, and
relevance to our work. Their notes are reproduced in full below the synthesis
(sections G1–G6). Every reader flags what it could not access; nothing
inaccessible was filled in.

Supersedes `audit-2026-08-24/03-literature-baselines-and-followons.md` and
`04-user-papers-folder.md` where they disagree; those were read mostly from
abstracts.

## Corrections made while reviewing (read these first)

- **Cola-GNN's arXiv ID is 1912.10202**, and that posting is the December
  2019 preprint ("Graph Message Passing with Cross-location Attentions for
  Long-term ILI Prediction"), not the CIKM 2020 camera-ready. Its Table 2/3
  gives Cola-GNN RMSE Japan 919 / 1060 / 1072 / 1156 / 1403 / 1500 and
  US-States 136 / 167 / 191 / 202 / 241 / 232 (h = 2/3/4/5/10/15), checked by
  reading the PDF page. The figures in our August audit and in EpiGNN's table
  (Japan 1051 / 1117 / 1372 / 1475 at h = 3/5/10/15) match none of the
  preprint's Japan values and most likely come from the CIKM version, which is
  paywalled and was not read. **G1's statement that EpiGNN copied Cola-GNN's
  numbers incorrectly is therefore not established**; it is a preprint versus
  published-version difference until the CIKM table is checked.
- **The "Wang 2023" survey the ledger quotes (E15, "merely focus on proximity,
  yet ignore the trend and periodicity") is Yi Wang (2023), International
  Journal of Digital Earth 16(1):2034–2066.** G6 could not identify it; G5 read
  it from the local folder, where the file is misnamed
  `Jun Zhao et al_2023_Advances in spatiotemporal graph neural network prediction research.pdf`.
- `doc/GNN forecasting/1-s2.0-S095741742502010X-main.pdf` is Yin et al. (2025),
  Expert Systems with Applications 291:128391, a process-mining paper on
  shipbuilding event logs, not epidemic forecasting. It is useful only as an
  Elsevier house-style template.
- **TERN's public code cannot run as published**: `src/train.py` line 27 does
  `from data import ColaGNNData`, and no `src/data/` exists in the repository
  (its `.gitignore` excludes every directory named `data/`). Verified with the
  GitHub API on 29 Sep 2026.
- Not accessed by any reader: STTGNN (KBS 2026, paywalled, no preprint),
  CNNRNN-Res (SIGIR 2018, paywalled), Cramer et al. PNAS 2022 (blocked), and
  the CIKM version of Cola-GNN. STTGNN is the most important of these to
  obtain through the library.

## Synthesis

### 1. What the field claims, and what supports it

| Paper | Venue | Claim | Evaluation | Significance test |
|---|---|---|---|---|
| Cola-GNN | CIKM 2020 | cross-location attention for long-horizon ILI | one model per lead h; mean/SD over 10 runs | no |
| EpiGNN | ECML-PKDD 2022 | transmission-risk graph learning; 5.6% gain | Cola-GNN row reported from the reference (asterisked) | no |
| STAN / MepoGNN | JAMIA 2021 / ECML-PKDD 2022 | graph attention; mobility + SIR | "L=15" / "7 days ahead" are averages over the whole output window | STAN only (bootstrap CI, t-tests) |
| HeatGNN, EARTH (ICML 2025), EpiHybridGNN | 2024–25 | SIR or neural-ODE hybrids | re-run baselines 45–50% worse than their own papers report (EARTH: Cola-GNN US-States h=5 299.1 vs 202) | no |
| PISID | PLOS ONE 2025 | graph-free region embeddings + SIR beat graph encoders | 5 seeds, mean ± SD; COVID-era data, not the ILI benchmark | no |
| TERN | arXiv 2609.18407 | delta-rule memory + seasonal reference + online adaptation matches seasonal-naive on Japan | lead-h, 50/20/30, horizon-averaged; copied Cola-GNN/EpiGNN rows | no |
| SpatialEpiBench, EpiCastBench, M-SPICE | 2026 | methods beat naive <50% of the time; adjacency can hurt; foundation models win where tested | rolling origin; Friedman + MCB (EpiCastBench) | partly |

- **Claimed gains are small and weakly evidenced.** Physics-informed hybrids
  gain 1.6–4%; spatial components add ~5–9% (M-SPICE, Mantis, GeoID-PINN).
  Our median minimum detectable effect is ~34% (ledger E18); no paper tests
  significance.
- **No paper inspects whether attention or the graph does anything.** Every
  paper argues from ablation deltas; none measures attention entropy or weight
  distributions. The KDD 2024 survey treats graph attention as unproblematic.
- **SIR/mechanistic hybrids are the crowded novelty route**: 5 of the 8
  successor papers bind SIR to a network.

### 2. The protocol problem is field-wide

- "Horizon" means three things: a single lead h (Cola-GNN, EpiGNN, TERN per
  horizon); an average over the whole output window (STAN, MepoGNN); an
  average across horizons (TERN's headline, SpatialEpiBench).
- Splits range over 60/20/20, 50/20/30, 6:1:3, and fixed 50-step windows in
  BDSTGNN's code, which contradict its paper.
- Re-implemented baselines come out far weaker than their original papers.
- A per-horizon, lead-h re-run of all five baselines on one platform exists in
  no paper.

### 3. The 2026 reviewer standard

Converging across the benchmark papers: (1) rolling-origin evaluation, (2) a
naive baseline at every horizon, (3) point and probabilistic scoring (WIS),
(4) a formal multi-model significance test (Friedman + MCB), (5) auditable
data and baseline provenance. No paper meets all five. None computes power or
a minimum detectable effect, none reports Cola-GNN/EpiGNN per horizon under a
corrected protocol, and none audits foundation-model pretraining corpora,
though TERN hard-codes a contamination flag for Chronos-2/TimesFM-3 on the US
sets. Bosse et al. (2023) recommend log-scale scoring as a complement to, not
a replacement for, natural-scale scoring. EpiLearn now ships conformal
intervals by default, so calibration is expected rather than novel.

### 4. How the papers are written

- Best template for CBM: **STAN** (structured abstract, bootstrap CIs and
  t-tests, a named limitations section). **PISID** for seed-variance
  reporting. **MepoGNN** for an honest limitations section that states a
  failure case.
- Elsevier house style (Yin et al. 2025, ESWA): numbered definitions before
  the method, a plain-language gloss after each equation, a two-part
  limitations section (model characteristics, data characteristics), and the
  CRediT/funding/data/competing-interest block straight after the conclusion.
- Patterns to avoid: one-sentence limitations; interpretability asserted from
  a single heatmap; "consistently outperforms" from averaged tables without
  variance; baseline rows pasted from other papers without saying so.

### 5. Consequences for our paper

1. Lead with evaluation: the cross-paper protocol inconsistency, baseline
   inflation, and the power analysis, which no paper provides.
2. The attention-collapse result is the first direct test of whether spatial
   attention in this family does anything.
3. SEER as the constructive part: graph-free (cite PISID), scale-equivariant,
   seasonal memory (cite TERN), a learned monotone horizon gate (TERN's is
   hand-set), evaluated on the daily COVID panels no competitor uses.
4. Add WIS and Friedman + MCB alongside the DM tests; report natural- and
   log-scale errors; drop "first calibrated forecaster".

---

## G1 — Foundations of the benchmark family (Cola-GNN, EpiGNN, CNNRNN-Res, STAN, MepoGNN)

> **Editor's note (29 Sep 2026).** This reader compared EpiGNN's Cola-GNN row against the arXiv preprint (1912.10202v2). The preprint is not the CIKM 2020 camera-ready, so the "mismatch" and the conclusion that the reference values came from EpiGNN's citation chain are not established; see *Corrections* at the top of this file.

Access log: Cola-GNN (arXiv 1912.10202v2, full 17pp PDF downloaded and read completely,
pages 1-17), EpiGNN (arXiv 2208.11517v1, full 16pp PDF read completely — this is the
ECML-PKDD 2022 paper's arXiv posting), STAN (local PDF
`doc/GNN forecasting/files/1624/Gao et al_2021_STAN.pdf`, full 11pp JAMIA paper read
completely), MepoGNN (arXiv 2306.14857v2, "Metapopulation Graph Neural Networks" — this
is the **extended journal version** of the ECML-PKDD 2022 paper, 13pp, read completely;
it explicitly states "Part of this work first published in [Vol. 13718, pp. 453-468, 2023]
by Springer Nature" — i.e. the original conference paper is Cao et al., ECML-PKDD 2022,
*Machine Learning and Knowledge Discovery in Databases*, pp. 453-468).
CNNRNN-Res (Wu, Yang, Nishiura, Saitoh, SIGIR 2018, pp. 1085-1088): **full text could
NOT be accessed.** It is a 4-page ACM short paper behind the ACM DL paywall (confirmed
403 Forbidden on direct fetch); no arXiv preprint exists; ResearchGate/Semantic Scholar
pages show only metadata, no OA PDF link. Everything below on CNNRNN-Res is drawn from
how it is *described and re-implemented* in the other four papers I did read in full
(Cola-GNN §4.3, EpiGNN §4.1, MepoGNN not used) plus their code — this is explicitly
second-hand and flagged as such throughout.

I also pulled the actual evaluation code from both public repos (via `gh api` +
raw.githubusercontent.com, not scraped through a summarizer) to check exactly how the
test metric is computed: `amy-deng/colagnn` (`src/data.py`, `src/train.py`) and
`Xiefeng69/EpiGNN` (`src/data.py`). Findings under each paper's section D and in the
dedicated Code section below.

---

### 1. Cola-GNN

#### A. Bibliographic
Songgaojun Deng (Stevens), Shusen Wang (Stevens), Huzefa Rangwala (George Mason), Lijing
Wang (UVA), Yue Ning (Stevens). "Graph Message Passing with Cross-location Attentions for
Long-term ILI Prediction." **CIKM 2020** (per STAN's citation and common knowledge; the
arXiv posting itself, v2 dated 29 Dec 2019, does not print a venue line — it is a
preprint). arXiv:1912.10202v2 [cs.LG]. Code: github.com/amy-deng/colagnn. Citation count
not independently verified (not pulled from a citation index in this session).

Correction of my starting assumption: the task brief suggested arXiv 2008.04436 "if that
is correct — verify" — **it is not correct**; 2008.04436 is an unrelated FPGA/Ising-model
paper. STAN's own reference list (ref. 9) gives the correct ID as **1912.10202**, which I
verified directly.

#### B. Claimed contributions (verbatim, from the paper)
> "We propose a novel graph-based deep learning framework for long-term epidemic
> prediction from a time-series forecasting perspective. This is one of the first works
> of graph neural networks adapted to epidemic forecasting."
> "We investigate a location-aware attention mechanism to capture location correlations.
> The influence of locations can be directed and automatically optimized in the model
> learning process. The attention matrix is further evaluated as an adjacency matrix in
> the graph neural network for modeling disease propagation."
> "We design a temporal convolution module to automatically extract temporal dependencies
> and hidden features for time-series data of multiple locations. The learned temporal
> features for each location are utilized as node attributes for the graph neural
> network."
> "The proposed method, Cola-GNN, outperforms a broad range of state-of-the-art models on
> three real-word datasets with different long-term prediction settings. We also
> demonstrate the effectiveness of its learned attention matrix compared to a geographical
> adjacency matrix in an ablation study."

#### C. Method (5-8 lines)
Per-location RNN (vanilla RNN — outperformed GRU/LSTM in their experiments) produces a
hidden state per location; a temporal-convolution branch (K 1D-CNN filters + max-pool)
produces a second, local temporal feature. "Location-aware attention" is *additive*
attention (Bahdanau-style) computed between every pair of locations' RNN hidden states,
row-normalized by an ℓp-norm (not softmax), then combined via a learned element-wise gate
M with the row-normalized geographic adjacency matrix Ã^g to give the final attention
matrix Â = M⊙Ã^g + (1−M)⊙A. Â is then used as the message-passing adjacency in a small
GNN operating on the temporal-conv features; RNN hidden state and GNN output are
concatenated and linearly mapped to the prediction. Loss is ℓ1 with weight-decay
regularization, Adam optimizer.

#### D. Data & protocol (exact quotes)
Three datasets — **Japan-Prefectures** (47×348, Aug 2012–Mar 2019), **US-Regions**
(10×785, 2002–2017, HHS ILINet), **US-States** (49×360, 2010–2017 CDC ILI, one state
dropped for missing data). Table 1 stats: JP min/max/mean/SD = 0/26635/655/1711; US-R =
0/16526/1009/1351; US-S = 0/9716/223/428.

Split, quoted exactly: **"After ordering the data by time, the first 50% is used for
training, next 20% for validation, and the last 30% for testing."** Chronological, no
shuffling. "The test data covers 2.1, 4.5, and 2.1 flu seasons in Japan-Prefectures,
US-States and US-Regions respectively... at least 3, 7.2 and 3 flu seasons in the three
training sets."

Input window: **20 weeks** ("roughly five months"). Horizons: short-term {2,3,4}, long-
term {5,10,15} (leadtime=1 explicitly excluded — "symptom monitoring data is usually
delayed by at least one week"). **Each horizon is its own trained model predicting only
that single lead h** — confirmed both in text ("the proposed method is not flexible
enough in the case that different models are trained for different lead time settings")
and directly in the released code (see Code section below): target `Y[i,:]` is read at
index `idx_set[i]`, with input window ending at `idx_set[i] - h + 1`, i.e. a single-step
target exactly h steps past the window end. **No pooling over a lead-time range.**

Normalization, quoted: **"Data is normalized to 0-1 range for each region. The maximum
value of the region is set to 1, and the minimum value of the region is set to 0."** The
paper text does not state explicitly whether max/min are fit on train-only or the full
series, but the released code fits them on the training split only (`self.max`/`self.min`
computed from `train_mx`, then applied to `self.rawdat`) — this is leakage-safe and
matches MSAGAT-Net's own protocol description in AGENTS.md.

Runs: **"All experimental results are the average of 10 randomized trials,"** and
Figures 2–3 show mean ± standard deviation across the 10 runs (error bars). No formal
significance test against baselines is reported in text (contrast with STAN, which does
run t-tests).

#### E. Baselines
GAR, AR, VAR, ARMA, RNN (vanilla, chosen over GRU/LSTM after tuning), RNN+Attn,
CNNRNN-Res (cited [31] = Wu et al. 2018 — description-only reimplementation),
GCNRNN-Res (Cola-GNN authors' own variant, swapping CNNRNN-Res's CNN module for a
2-layer GCN using the geographic adjacency matrix). All baselines appear to be re-run
by the Cola-GNN authors themselves (own parameter search, own runtime/parameter-count
table, Table 5) rather than numbers copied from the CNNRNN-Res paper.

#### F. Headline numbers (verified directly from Table 2/3 of the PDF, not summarized)
RMSE, Cola-GNN, by horizon 2/3/4 (Table 2) and 5/10/15 (Table 3):
- **Japan-Prefectures**: 919 / 1060 / 1072 / 1156 / 1403 / 1500
- **US-Regions**: 483 / 633 / 765 / 871 / 1126 / 1218
- **US-States**: 136 / 167 / 191 / 202 / 241 / 232

PCC, Cola-GNN, same horizon order:
- JP: 0.911 / 0.893 / 0.894 / 0.883 / 0.818 / 0.754
- US-R: 0.944 / 0.905 / 0.863 / 0.832 / 0.719 / 0.639
- US-S: 0.955 / 0.933 / 0.907 / 0.897 / 0.822 / 0.859

Strongest baseline at long horizons is GCNRNN-Res (JP h=10 RMSE 1384, best in table,
bold; US-S h=15 PCC 0.814, close to Cola-GNN's 0.859).

**Cross-check against the task brief's reference numbers** ("Cola-GNN's own reported
RMSE — Japan 929/1051/1117/1372/1475, US-Regions 480/636/855/1134/1203, US-States
136/167/202/241/237 for horizons 2/3/5/10/15"): these do **not** exactly match what
Cola-GNN's own paper actually reports (I read the table directly — see numbers above).
The brief's numbers are close but consistently *lower* (better) than Cola-GNN's true
self-reported values at h=3/5/10/15 for Japan and US-Regions. I traced the source: these
are **EpiGNN's** reproduction of Cola-GNN as a baseline (see §2.F below) — EpiGNN marks
that row with an asterisk claiming "the result is reported in the corresponding
reference," but the asterisked numbers still don't match Cola-GNN's own paper. This is a
real, documentable inconsistency in the baseline-number lineage across the family (see
Synthesis).

#### G. Ablations / interpretability
Table 4: "Cola-GNN w/o temp" (drop temporal-conv, feed raw series into message passing)
and "Cola-GNN w/o loc" (drop location-aware attention, use only geographic adjacency).
Full model wins at long horizons on JP/US-R; on US-States the ablated variants are
sometimes *slightly better* (paper concedes this: "In the US-states dataset, models
without temporal or location-aware attention modules are sometimes slightly better than
the full model... adding temporal and spatial modules does not change the short-term
prediction very much"). Attention visualization (Fig. 6-7): shows the learned attention
matrix Â differs visibly from raw geographic distance and from raw input correlation,
and a qualitative case study (region 5 in US-Regions) where "high attention" neighbors
share an earlier outbreak onset than "low attention" neighbors — a single qualitative
example, not a systematic entropy/collapse check.

#### H. Writing
Sections: Abstract, Intro, Related Work (2 subsections), Proposed Method (6
subsections incl. Algorithm-1 pseudocode box), Experiment Setup (dataset/metrics/
baselines/hyperparameters), Results (6 subsections: prediction performance, case
studies, attention visualization, ablation, sensitivity analysis ×3, model complexity),
Conclusion, References. ~17 pages incl. references and many figures (bar charts w/
error bars, line plots, heatmaps of attention/correlation/geolocation matrices side by
side). No explicit "Limitations" section; the Conclusion contains one paragraph of
limitations (inflexibility across lead times; ignores external features like weather/
migration) framed as "future work," not as caveats on the claims made. No formal
reproducibility statement / no code link printed inside the PDF itself (code lives only
on GitHub, found externally). Novelty argued primarily against non-graph deep baselines
(RNN, RNN+Attn) and against CNNRNN-Res/GCNRNN-Res, not against other GNN epidemic work
(EpiGNN, MepoGNN, STAN all postdate it).

#### I. Red flags
- Paper prose is silent on whether normalization is fit on train-only or the whole
  series; only the code confirms it's train-only (favorable finding, but the paper
  itself doesn't state it — a reader taking only the paper at face value could suspect
  leakage).
- Ablation shows the "full model" is *not* uniformly best (loses to ablated variants on
  US-States) but the abstract/contributions still claim the method "outperforms a broad
  range of SOTA models" unqualified.
- Runtime/parameter Table 5 measured on US-States only (largest N) and extrapolated as
  representative; not shown per-dataset.
- No significance testing against baselines (unlike STAN, which explicitly runs t-tests).

#### J. Relevance to the new paper
Cola-GNN is the *origin* of the location-aware/cross-location attention idea that
MSAGAT-Net's EAGAM module descends from, and its own ablation already shows attention
adds little on the smallest, least noisy dataset (US-States) — an early, easy-to-miss
precedent for "attention may not be doing real work," directly relevant to claim (ii)
(attention collapse). Its single-lead-per-model protocol (confirmed in code) is the
*correct* baseline against which to detect the pooled-metric bug the brief warns about —
Cola-GNN itself does not have that bug. Its train-only normalization is a good citation
for "this is/is not how it should be done."

---

### 2. EpiGNN

#### A. Bibliographic
Feng Xie, Zhong Zhang, Liang Li, Bin Zhou (corresponding), Yusong Tan — College of
Computer, National University of Defense Technology. "EpiGNN: Exploring Spatial
Transmission with Graph Neural Network for Regional Epidemic Forecasting." **ECML-PKDD
2022**, in *Machine Learning and Knowledge Discovery in Databases*, DOI
10.1007/978-3-031-26422-1_29 (per DBLP/ACM search results; page range not stated inside
the arXiv PDF I read). arXiv:2208.11517v1 [q-bio.QM], posted 23 Aug 2022. Code + data:
github.com/Xiefeng69/EpiGNN (link given verbatim in the paper's own intro). Citation
count not independently verified in this session.

#### B. Claimed contributions (verbatim)
> "We design a novel graph neural network-based model for epidemic prediction in which a
> transmission risk encoding module is proposed that shows how we incorporate local and
> global spatial effects of regions into the model."
> "We introduce a Region-Aware Graph Learner which takes transmission risk, geographical
> information, and temporal dependencies into account to better explore underlying
> spatio-temporal correlations between regions."
> "We evaluate our model on five epidemic-related datasets. Experimental results show the
> proposed method achieves state-of-the-art performance and demonstrate the effectiveness
> of our model. The source code and datasets are available at
> https://github.com/Xiefeng69/EpiGNN."
Abstract also states the headline number: **"EpiGNN outperforms state-of-the-art
baselines by 9.48% in RMSE"** (this is an average across all tasks, not a per-dataset
figure — see F).

#### C. Method (5-8 lines)
Multi-scale dilated 1D convolutions (several filter sizes/dilations in parallel,
concatenated + adaptive-pooled) give a temporal feature h^temp per region. A
"transmission risk encoding" module computes (i) a Local Transmission Risk (LTR) = a
linear map of each node's degree in the geographic adjacency graph, and (ii) a Global
Transmission Risk (GTR) = a self-attention-style correlation over h^temp, summed per row
and linearly mapped. These three features (h^temp, h^l, h^g) sum to a node feature
h^feat. A "Region-Aware Graph Learner" (RAGL) builds an *asymmetric* correlation matrix
from h^temp via a subtraction-based dot-product (tanh(M1M2ᵀ − M2M1ᵀ), ReLU'd, explicitly
designed to break the usual attention symmetry), gated by a degree-product term, added to
the geographic adjacency and optionally an external-resource term (e.g. mobility). This
feeds a GCN; GCN output is concatenated with h^feat and an optional linear AR component,
summed to the final prediction. Trained with plain MSE loss.

#### D. Data & protocol (exact quotes, cross-checked against Table 1 in the PDF)
Five datasets — the same **Japan-Prefectures (47×348)**, **US-Regions (10×785)**,
**US-States (49×360)** as Cola-GNN (identical min/max/mean/SD in Table 1, confirming
shared data), plus two COVID datasets: **Australia-COVID** (8 regions×556 days, daily,
JHU-CSSE) and **Spain-COVID** (35 NUTS3 regions×122 days, daily, incl. Facebook "Data For
Good" mobility as an optional external signal).

Split, quoted: **"All datasets have been split into training set (50%), validation set
(20%), and test set (30%) in chronological order."** — identical wording/fractions to
Cola-GNN; confirmed in code (see Code section) to be the *literal same* `DataBasicLoader`
logic, including the same train-only max/min normalization.

Input window T = 20. Horizons: influenza {3,5,10,15}, COVID {3,7,14}. **Each horizon is
a separately trained model predicting a single lead h** (confirmed both by the paper's
Table 2/3 layout, which reports each h as its own column with its own baseline row, and
directly in the code, which is essentially a fork of Cola-GNN's `data.py` — same
`Y[i,:] = self.dat[idx_set[i], :]` single-index target construction, same `end = idx -
h + 1` window-end logic). No pooling over a range of leads.

Runs: **"For each task we run 5 times with different random initialization."** No
standard deviations or significance tests reported anywhere in the paper (contrast:
Cola-GNN reports SD via error-bar figures; STAN reports 95% CIs and explicit t-tests).
Implementation: Python 3.8.5, PyTorch 1.9.1, CUDA 11.1, single Nvidia K80 GPU.

#### E. Baselines
HA, AR, LSTM, TPA-LSTM, ST-GCN, CNNRNN-Res, SAIFlu-Net, Cola-GNN. **Table 2 marks the
Cola-GNN row with an asterisk (*) meaning "the result is reported in the corresponding
reference"** — i.e. EpiGNN states explicitly that it did **not** re-run Cola-GNN, but
copied its numbers from elsewhere. Crucially, as noted under Cola-GNN §F, **the copied
numbers do not match Cola-GNN's own paper's self-reported table** (EpiGNN's Cola-GNN
row: JP RMSE h3/5/10/15 = 1051/1117/1372/1475, PCC h10 marked with its own underline
0.813*; my direct read of Cola-GNN's Table 2/3 gives 1060/1156/1403/1500). Every other
baseline (HA, AR, LSTM, TPA-LSTM, ST-GCN, CNNRNN-Res, SAIFlu-Net) has no asterisk in
Table 2, implying EpiGNN re-ran/reimplemented those itself.

#### F. Headline numbers (verified directly from Table 2/3 of the PDF)
RMSE / PCC, EpiGNN, horizon 3/5/10/15:
- **Japan-Prefectures**: RMSE 996/1031/1441/1470; PCC .904/.908/.739/.773
- **US-Regions**: RMSE 589/774/984/1061; PCC .912/.842/.749/.694
- **US-States**: RMSE 160/186/220/236; PCC .935/.907/.865/.861

Cola-GNN* (asterisked/copied) same order: JP 1051/1117/1372/1475; US-R 636/855/1134/
1203; US-S 167/202/241/237. (These match the task brief's reference numbers exactly —
confirming the brief's reference values were themselves sourced from EpiGNN's table, not
from Cola-GNN's own paper.)

COVID (Table 3), RMSE at horizon 3/7/14: Spain-COVID EpiGNN 135.54/162.51/186.41 vs
EpiGNN_exter (with mobility) 129.90/145.33/178.73 vs Cola-GNN 138.34/176.52/203.67;
Australia-COVID EpiGNN 71.42/153.07/287.90 vs Cola-GNN 127.59/279.56/326.79.

At US-States h=15, EpiGNN (236) barely beats Cola-GNN* (237) — a ~0.4% RMSE difference,
essentially noise given no variance is reported, despite the abstract's blanket "9.48%"
improvement claim (which is an average, and evidently pulled up by the COVID datasets
where the margin is much larger, e.g. Australia h=3: 71 vs 128, an ~44% cut).

#### G. Ablations
Fig. 3: w/o LTR, w/o GTR, w/o RAGL (RAGL replaced by plain softmax self-attention), run
on **Japan-Prefectures and US-States only** — text says "we perform ablation studies on
Japan-Prefectures and US-Regions datasets" but the figure caption says "US-States (top)
and Japan-Prefectures (bottom)" — an internal inconsistency between prose and figure
caption (US-Regions vs US-States); neither COVID dataset nor US-Regions actually appears
in Fig. 3. All three ablations degrade performance; RAGL specifically outperforms plain
self-attention, which the authors attribute to avoiding "oversmoothing" from
bidirectional/symmetric attention. Parameter sensitivity (Fig. 4: filter count k, GCN
layer count l) run only on US-Regions/US-States, not Japan or COVID.

#### H. Writing
Sections: Abstract, Intro (with a 3-panel schematic distinguishing geographic topology
vs local vs global transmission effects — a nice framing device), Related Work (2
subsections: statistical/mechanistic, GNN-based), Proposed Method (6 subsections +
Algorithm-1 box), Experiments and Analysis (5 subsections: settings, prediction
performance, ablation, parameter analysis, visualization), Conclusions, References.
16 pages (the arXiv/extended posting; the camera-ready ECML-PKDD version is reportedly
~8 pages per DBLP, i.e. arXiv carries extra appendix-style content: parameter
sensitivity, extra visualizations, runtime/param-count table). Runtime+parameter-count
table (Table 4) directly modeled on Cola-GNN's Table 5. No explicit "Limitations"
section — future work is one sentence in the Conclusion ("we will devote to better
predict by considering the time decay effects of spatial transmission"). Figures include
US-map choropleths of learned degree/correlation, and a Texas-centric correlation map —
a good interpretability figure type worth imitating. Positions itself directly against
Cola-GNN (symmetric/bidirectional attention → oversmoothing) as its main foil.

#### I. Red flags
- **Copied baseline numbers that don't match the source** (see E/F) — a real citation-
  chain problem worth flagging explicitly if a future paper cites "Cola-GNN's numbers"
  via EpiGNN's table.
- Headline "9.48% average RMSE improvement" is a blended average across five very
  different datasets/horizons/scales; at the hardest long-horizon influenza setting
  (US-States h=15) the margin is <1%, which the abstract does not disclose.
- No variance/significance reporting despite running 5 seeds — the 5 runs are averaged
  but SD is never shown, so none of the RMSE gaps (e.g., EpiGNN 996 vs 2nd-best
  underlined value at JP h=3) can be assessed for significance.
- Ablation coverage is partial (2 of 5 datasets) and the text/figure-caption mismatch
  (US-Regions vs US-States) suggests a copy-paste error that was never caught in review.
- Split fractions/protocol are a near-verbatim copy of Cola-GNN's, including the same
  potential ambiguity about whether the *paper text* (as opposed to the code) discloses
  train-only normalization — it doesn't, for either paper.

#### J. Relevance to the new paper
Directly useful precedent for claim (ii): EpiGNN explicitly builds RAGL to *avoid*
softmax self-attention's oversmoothing/symmetry problems and shows in ablation that an
asymmetric, ReLU-gated formulation beats plain self-attention — i.e., even a paper in
this family independently found that naive attention degrades to something too uniform
to be useful, which supports (from a different angle) the claim that learned graph
attention in this family tends toward collapse. The exact code-level match between
EpiGNN's and Cola-GNN's `data.py` is strong, citable evidence that this family shares one
evaluation harness (single-lead target, train-only normalization, chronological 50/20/30
split) — useful for describing "the corrected protocol" as continuous with, not a
rejection of, the family's better practices, while still flagging the copied-number
discrepancy as a documented lineage problem.

---

### 3. CNNRNN-Res

**Full text not accessible.** Wu, Yuexin; Yang, Yiming; Nishiura, Hiroshi; Saitoh, Masaya.
"Deep Learning for Epidemiological Predictions." *SIGIR 2018* (41st Intl ACM SIGIR
Conf.), pp. 1085–1088 (a 4-page short paper). DOI 10.1145/3209978.3210077. I attempted:
direct WebFetch of the ACM DL page (403 Forbidden — paywalled), Semantic Scholar page
(no OA PDF field populated), ResearchGate (request-PDF only, not open), Semantic Scholar
API (rate-limited 429 on retry), Yiming Yang's CMU publications page (503 Service
Unavailable on fetch), and general web search for a mirror/preprint (none found — no
arXiv posting exists for this paper). **I could not read the original text and did not
fabricate any details about it.** Author list itself has a minor discrepancy across
secondary sources: most citations (ACM, and Cola-GNN/EpiGNN/MepoGNN's own reference
lists, which I did read directly) list **Wu, Yang, Nishiura, Saitoh**; one search snippet
attributed to a CMU publications listing suggested "Wu, Yang, Liu" (Hanxiao Liu) instead
of Nishiura/Saitoh — I was not able to verify the CMU page directly (fetch failed), so I
cannot resolve this discrepancy and flag it as unverified rather than asserting either
version.

Everything below is second-hand, drawn only from how CNNRNN-Res is described in the four
papers I did read cover-to-cover, and is marked as such.

- **Method, as described by Cola-GNN**: "CNNRNN-Res [31] A deep learning framework that
  combines CNN, RNN, and residual links to solve epidemiological prediction problems."
  As described by EpiGNN: "a deep learning model that combines CNN, RNN, and residual
  links for epidemiological prediction." General web-search summaries (not primary,
  treat with caution) describe an adjacency-shaped CNN filter over neighboring regions
  feeding an RNN, with residual connections added to combat overfitting — consistent
  with, but not confirmable beyond, the one-line descriptions in Cola-GNN/EpiGNN.
- **Role in the benchmark family**: it is the *first* deep-learning epidemic forecasting
  paper cited by every other paper in this group (Cola-GNN ref [31], EpiGNN ref [17],
  MepoGNN does not cite it directly) and is uniformly used as a weak/mid-tier baseline,
  never as SOTA, in every table I read: Cola-GNN Table 2/3 (CNNRNN-Res RMSE always well
  behind Cola-GNN — e.g. JP h=15 1862 vs Cola-GNN's 1500), EpiGNN Table 2 (JP h=15 RMSE
  1862 — **identical number to Cola-GNN's own CNNRNN-Res row**, i.e. EpiGNN reused
  Cola-GNN's CNNRNN-Res numbers verbatim without an asterisk, another instance of
  unlabeled number reuse across this family), EpiGNN Table 4 (smallest parameter count
  of any baseline: 13K/5K/14K on JP/US-R/US-S, vs EpiGNN's own 11K/9K/12K).
- Given full inaccessibility, I cannot report its own claimed contributions, its own
  data/protocol details (dataset, split, horizon handling), its own ablations, or its own
  writing style — only that it is consistently treated as a "does not use spatial
  correlation well" baseline that both graph-based successors (Cola-GNN, EpiGNN) beat by
  a wide and growing margin as horizon increases.

#### J. Relevance to the new paper
Because CNNRNN-Res is the historical zero-point ("first deep learning model for this
task") against which every graph-based improvement in this literature is measured, and
because I found at least one instance of a baseline number (its own JP h=15 RMSE) being
silently reused verbatim from one paper to the next without acknowledgment, it is worth
noting in the related-work section as evidence that number provenance in this literature
is not consistently tracked — relevant background for the new paper's protocol-correction
framing (i), even though I cannot personally verify CNNRNN-Res's own claims.

---

### 4. STAN

#### A. Bibliographic
Junyi Gao, Rakshith Sharma, Cheng Qian, Lucas M. Glass, Jeffrey Spaeder, Justin Romberg,
Jimeng Sun, Cao Xiao (corresponding). "STAN: spatio-temporal attention network for
pandemic prediction using real-world evidence." *Journal of the American Medical
Informatics Association (JAMIA)*, 28(4), 2021, pp. 733–743. doi:10.1093/jamia/ocaa322.
Received 22 Jul 2020, accepted 2 Dec 2020, published online 22 Jan 2021. Open access
(CC-BY 4.0). Code: github.com/v1xerunt/STAN (given in-text). Read from the local PDF
in full (11 pages incl. references).

#### B. Claimed contributions / abstract (verbatim, key lines)
> "Objective: We aim to develop a hybrid model for earlier and more accurate predictions
> for the number of infected cases in pandemics by (1) using patients' claims data from
> different counties and states that capture local disease status and medical resource
> utilization; (2) utilizing demographic similarity and geographical proximity between
> locations; and (3) integrating pandemic transmission dynamics into a deep learning
> model."
> "Results: STAN outperforms traditional epidemiological models such as SIR, SEIR, and
> deep learning models on both long-term and short-term predictions, achieving up to 87%
> reduction in mean squared error compared to the best baseline prediction model."

#### C. Method (5-8 lines)
Builds an attributed graph over counties/states with edges weighted by
population/geographic-distance gravity model (w_ij ∝ p_i^α p_j^β exp(−d_ij/r)); a
2-layer multi-head GAT (standard Veličković-style, softmax attention) produces node
embeddings from static (lat/lon/pop/density) + dynamic (active cases, hospitalizations,
ICU stays, 48 CDC-derived ICD-code counts from IQVIA claims data) features; embeddings
are max-pooled across the graph per timestep and fed to a GRU; the GRU's final hidden
state feeds two heads — (1) a direct MLP prediction of ΔI, ΔR for the whole prediction
window (multi-task short-term loss), and (2) an MLP predicting SIR parameters β, γ per
prediction window, which are then propagated forward via the discrete SIR difference
equations to give a second, "dynamics-constrained" set of ΔI, ΔR predictions used only as
an *auxiliary loss term* (not fed back into the primary output) to regularize the GRU's
hidden representations toward physically plausible trajectories.

#### D. Data & protocol (exact quotes)
US county-level (193 counties, "with more than 1000 confirmed cases by May 17") and
state-level (45 states) COVID-19 data, two sources merged: JHU Coronavirus Resource
Center (cases/deaths, Mar 22–Jun 10 2020) and **IQVIA claims data** (hospital/ICU
visits + frequency of 48 COVID-related ICD-10 codes per county/day; "records for a total
of 453,089 patients"). Input window L_I = **5** days (fixed, unusually short vs the
20-week windows in Cola-GNN/EpiGNN — reflects COVID's much faster dynamics and daily vs
weekly granularity). Prediction windows L_P ∈ {5, 15, 20} days.

Split, quoted: **"All training sets start from March 22, and all test sets start from
May 17. We also split L_P days from the training sets as validation sets... All
locations are used in training and testing set by splitting along the time dimension,
where early time windows are used for training, and later time windows are used for
testing the model."** Chronological, single fixed cutoff (not a moving/rolling split);
same cutoff date used regardless of L_P.

**Multi-step handling**: STAN predicts the **entire L_P-day window in one shot** via a
vector output ΔÎ, ΔR̂ ∈ ℝ^{L_P} from a single MLP applied to the GRU's last hidden state —
i.e. it is explicitly a *direct multi-horizon* method, unlike Cola-GNN/EpiGNN's
single-lead-per-model approach. The evaluation metric (MSE/MAE/CCC) is then computed
"for both long-term and short-term predictions" at each L_P setting, and Tables 1-2 report
one aggregate MSE/MAE/CCC number **per L_P value**, which — unlike Cola-GNN/EpiGNN's
per-single-horizon tables — is **pooled over the entire L_P-day window** rather than
reported at one specific lead time. This is an important, easily-missed protocol
difference from the Cola-GNN/EpiGNN family: STAN's Table 1/2 numbers for "L_P=15" are an
average error over days 1-15 ahead, not the error at day 15 specifically. A reader
comparing STAN's MSE at "L_P=15" against Cola-GNN's RMSE at "horizon=15" would be
comparing a windowed/pooled metric against a single-lead metric — not apples to apples.
(Confirmed by re-reading the Prediction section: "we can obtain longer predictions by
using a larger prediction window L_P.")

Normalization method not stated explicitly in the read pages (not addressed as its own
subsection — the paper focuses instead on the graph/feature construction).

Runs/CI: **"To estimate a 95% confidence interval... we resample the locations 1000
times, calculate the score on the resampled sets, and then use the 2.5 and 97.5
percentiles... as our confidence interval estimate."** This is a bootstrap-over-locations
CI, not a multi-seed training CI — i.e. it captures cross-location variance in a single
trained model's errors, not training-run-to-run variance. **"We conducted a T-test
between STAN and each baseline model to check the performance difference statistically.
The results show that for each baseline model, STAN can significantly outperform
statistically (P value < .001)."** Exact p-values given in Table 3 (e.g. GRU 5.27E-16 at
State-5). This is the most rigorous significance-testing practice of the five papers.

#### E. Baselines
SIR, SEIR (both re-fit per location via differential-equation fitting, not deep-learning
re-implementations), GRU, **ColaGNN** (re-run — "ColaGNN uses a location graph to extract
spatial relationships for predicting pandemics. Different from STAN, graph nodes in
ColaGNN only consist of time series of number of cases"), CovidGNN (Kapoor et al. 2020,
re-run — GNN with skip connections, no RNN). Plus two ablations used as baselines:
STAN-PC (removes transmission-dynamics loss) and STAN-Graph (removes GNN/graph
entirely). All baselines appear re-implemented/re-run on STAN's own IQVIA+JHU dataset
(not copied numbers from Cola-GNN's own paper, since Cola-GNN never used COVID county
data) — code repo cited (v1xerunt/STAN).

#### F. Headline numbers (verified from Table 1/2 of the PDF directly)
State-level (Table 1), STAN MSE / MAE / CCC at L_P=5/15/20:
- L_P=5: MSE 237,412; MAE 220.50; CCC 0.84 (best baseline ColaGNN: MSE 601,840, MAE
  440.26, CCC 0.66)
- L_P=15: MSE 972,192; MAE 586.56; CCC 0.84 (ColaGNN: MSE 7,192,031, MAE 1290.41,
  CCC 0.57)
- L_P=20: MSE 4,909,604; MAE 1088.48; CCC 0.82 (ColaGNN: MSE 9,317,132, MAE 1645.42,
  CCC 0.63)

County-level (Table 2), STAN at L_P=5/15/20: MSE 44,177/157,243/326,258; MAE
79.80/193.85/253.86; CCC 0.66/0.72/0.71 (vs ColaGNN MSE 61,627/465,104/703,377).

Text-stated aggregate improvements: "When L_P=5, STAN achieves 59% lower MSE, 33% lower
MAE, and 23% higher CCC than the best baseline ColaGNN. When L_P=15, STAN achieves 87%
lower MSE... When L_P=20, STAN achieves 48% lower MSE..." (state-level); analogous 26%/
55%/55% MSE reductions at county level.

**Note on comparability**: because these are COVID-19 case-count MSE/MAE on a totally
different scale, population, and time period than Cola-GNN's/EpiGNN's ILI RMSE numbers,
and because STAN's ColaGNN baseline is re-trained on STAN's own dataset (not the
Japan/US-Regions/US-States ILI data), **none of these numbers are directly comparable to
the Cola-GNN/EpiGNN reference values** given in the task brief — that comparison would be
a category error. This is itself worth stating plainly in any related-work table.

#### G. Ablations / interpretability
STAN-PC (no dynamics-constraint loss) and STAN-Graph (no GNN/graph) both underperform
full STAN but both still beat all other baselines in Tables 1-2 — i.e., graph structure
and the physics-informed loss each contribute, and the paper explicitly claims "both
reduced model[s]... outperform other baselines. This indicates that both transmission
dynamics constraints and real-world evidence provide valuable information." No attention-
entropy or attention-collapse diagnostic of any kind is run — STAN never checks whether
its GAT attention is doing anything beyond uniform pooling; the "graph structure helps"
claim rests entirely on the STAN-Graph ablation's higher error, not on any inspection of
what the attention weights actually converged to.

#### H. Writing
Structure: Abstract (Objective/Materials & Methods/Results/Conclusions — structured
medical-journal abstract, JAMIA house style), Introduction (2 subsections: prior
epidemic-prediction models, prior physics-informed GNNs), Objective, Materials and
Methods (Problem formulation, Graph construction, Modeling spatio-temporal patterns via
GAT, Modeling temporal features via RNN, Multitask prediction and transmission-dynamics
loss, Prediction with STAN), Experiments (dataset description, baseline models, tasks
and evaluation strategy), Results, **Discussion and Limitations** (explicit named
section — the only one of the five papers to have one), Conclusion, Funding, Author
Contributions, Conflict of Interest, Data Availability, Supplementary Material,
References. 11 pages. Data-availability statement explicitly names both data sources and
states the claims data are "shared on request." Explicit, itemized limitations: (1)
fixed prediction-window setting rather than iterative/rolling, requiring re-training for
new windows; (2) dynamics constraints "may be too simple to reflect real-world
situations, such as home isolation and pandemic control policies"; (3) data-quality/ICD-
code-lag issues in the claims data used to build the attributed graph. This is
meaningfully more self-critical writing than Cola-GNN, EpiGNN, or MepoGNN, and its
structured medical-journal abstract + named limitations section is a strong template to
imitate for a JAMIA-adjacent or CBM-target evaluation paper.

#### I. Red flags
- No attention-collapse or entropy diagnostic despite attention being one of the paper's
  three headline contributions — the "graph helps" claim is inferred solely from
  ablation MSE deltas, never from inspecting what the GAT actually learned (contrast
  MSAGAT-Net's own finding, ledger E3/E7, that this exact failure mode — inert/near-
  uniform spatial attention — occurs in a closely related architecture).
- Pooled-window metric (MSE over the full L_P-day window) is reported under a label
  ("L_P=15") that reads, at a glance, like a single-horizon number the way Cola-GNN/
  EpiGNN report theirs — this is exactly the kind of ambiguity the task brief warned
  about, though in STAN's case it's disclosed in the method text (Fig. 2 makes clear
  L_P is a window), just easy to conflate with the family's convention if skimmed.
  I did not find evidence STAN itself made a scoring *error*, only that its protocol is
  structurally different from Cola-GNN/EpiGNN's in a way that is easy to mis-read as
  equivalent.
- Test window is very short (~24 days, May 17–Jun 10 2020) and single-cutoff, covering
  only the initial 2020 wave — generalization across later, more heterogeneous COVID
  waves is untested.
- "Up to 87% reduction in MSE" (abstract) is the single best-case number among six
  MSE/MAE/CCC × 2 granularity × 3 window combinations, not a typical or median
  improvement — classic best-case headline selection.

#### J. Relevance to the new paper
Best template in this group for the target CBM-style evaluation paper's structure
(structured abstract, named Discussion-and-Limitations section, explicit bootstrap CI +
t-test practice — directly relevant to claim (iv), power analysis / calibrated
intervals). Also the clearest illustration that "graph attention helps" claims in this
literature are almost always argued via ablation-delta rather than direct inspection of
attention weights — exactly the gap claim (ii) is meant to close. Its pooled-window
metric convention is a second, independent example (alongside the brief's original
concern about a Cola-GNN fork) of how "horizon" can mean different things across papers
in ways that break naive cross-paper RMSE comparisons — good supporting evidence for the
protocol-correction framing (i).

---

### 5. MepoGNN

#### A. Bibliographic
Original conference paper: Qi Cao, Renhe Jiang (corresponding), Chuang Yang, Zipei Fan,
Xuan Song, Ryosuke Shibasaki. "MepoGNN: Metapopulation Epidemic Forecasting with Graph
Neural Networks." **ECML-PKDD 2022**, in *Machine Learning and Knowledge Discovery in
Databases*, Springer Nature Switzerland, 2023, **pp. 453-468** (LNAI vol. 13718) — this
exact citation is given by the extended paper's own reference [1], which I read directly.
Extended journal version (what I actually read in full): "Metapopulation Graph Neural
Networks: Deep Metapopulation Epidemic Modeling with Human Mobility," arXiv:2306.14857v2
[cs.CY], 13 pages, explicitly marked "†This is the extended version of the ECML-PKDD2022
paper [1]. The main incremental changes: a mobility generation method and the experiments
to test its effectiveness, the detailed descriptions of data processing, the
visualization and analysis of data, the limitation discussion based on extra test data."
Code: github.com/deepkashiwa20/MepoGNN (per external search; not printed inside the PDF
itself in the pages I read, though the repo does exist and matches the paper's method
names — Main.py, adaptive/dynamic graph learning variants).

#### B. Claimed contributions (verbatim)
> "We propose a novel hybrid model along with two types of graph learning module for
> multi-step multi-region epidemic prediction by mixing metapopulation epidemic model and
> spatio-temporal graph convolution networks."
> "Our model can explicitly learn the time/region-varying epidemiological parameters as
> well as the latent epidemic propagation among regions from the heterogeneous inputs
> like infection related data, human mobility data, and meta information in a completely
> end-to-end manner."
> "We collect and process the big human GPS trajectory data and other COVID-19 related
> data that covers the 47 prefectures of Japan from 2020/04/01 to 2021/09/21 for
> countrywide epidemic forecasting."
> "We conduct comprehensive experiments... by comparing with three classes of baseline
> models. The results illustrate the superior forecasting performance of our model,
> especially for unprecedented surge of cases."
> "We present a mobility generation method with minimal data requirement to handle the
> situation which mobility data is unavailable."
Also states explicitly: **"To the best of our knowledge, our work is the first hybrid
model that couples metapopulation epidemic model with spatio-temporal graph neural
networks."**

#### C. Method (5-8 lines)
Extends the classical two-population metapopulation SIR (S,I,R per prefecture, coupled
by a human-mobility term h_nm between every prefecture pair n,m) so that β, γ, and the
propagation graph H are all **time- and region-varying, predicted by a neural network**
rather than fit/fixed. A spatio-temporal module (stacked "ST layers" combining Gated TCN
+ diffusion GCN, à la GraphWaveNet, with gated dense connections between layers) consumes
node features + a weighted adjacency A and outputs the predicted β^{t+1..T_out} and
γ^{t+1..T_out} sequences via two separate FC heads. A graph-learning module produces a
**single shared learnable graph A** used both by the spatio-temporal module and by the
metapopulation-SIR propagation term itself (the paper's key structural choice, meant to
keep the two components consistent/interpretable) — offered in two variants: "Adaptive"
(static commuter-survey flow matrix as an initialization for a fully learnable graph) or
"Dynamic" (a learnable time-lag-weighted average of the actual daily OD mobility flow
tensor). The metapopulation-SIR module then iterates S,I,R forward T_out steps using the
predicted β, γ, H to produce the final daily-confirmed-case output — i.e., unlike every
other model in this group, **the model's output is mechanistically constrained to be the
output of an SIR-type recurrence**, not a free MLP/GNN readout.

#### D. Data & protocol (exact quotes)
Single country/domain: **47 prefectures of Japan, 2020/04/01–2021/09/21 (539 days)**,
daily confirmed cases (NHK COVID-19 database) + recovered (Japan LIVE Dashboard/MHLW) +
population (2020 census) + external features (Facebook Movement Range Maps movement
change, ratio of active/confirmed cases, day-of-week) + mobility (static: 2015 census
commuter-survey OD matrix; dynamic: GPS-derived daily OD flow tensor from Blogwatcher
Inc. mobile data, explicitly bias-corrected for sampling-rate imbalance via a
stay-put-ratio normalization the paper derives itself, Eqs. 17-19). This is a
fundamentally different dataset family from Cola-GNN/EpiGNN/STAN — no Japan-Prefectures-
ILI, US-Regions, or US-States comparison is attempted anywhere in the paper.

Split, quoted: **"we split the data with ratio 6:1:1 to get training/validation/test
datasets."** (i.e. 75%/12.5%/12.5%, not the family's usual 50/20/30 — a materially
different split fraction.) Not explicitly stated as chronological in that sentence, but
the surrounding text ("the fifth wave of infection in Japan is included in test dataset
to test the model performance in a real outbreak situation") confirms the test split is
the most recent time period, i.e. chronological by construction. Curriculum-learning
training strategy explicitly used: **"we increase one prediction horizon every two
epochs starting from one day ahead prediction until reaching output time length."**

Input/output: T_in = T_out = **14 days** (fixed at both ends — "two-week historical
observations to do the two-week prediction of daily confirmed cases"). **Explicitly
multi-step, direct (not autoregressive/iterative) output of all 14 days at once** via
the SIR-module's forward-iteration of the *predicted parameter sequences* β^{1..14},
γ^{1..14} — structurally similar to STAN's "one-shot vector output" approach, and
unlike Cola-GNN/EpiGNN's separate-model-per-single-lead-h approach. Table I reports
metrics at "3 Days Ahead," "7 Days Ahead," "14 Days Ahead" **and "Overall"** — the
"Overall" column is explicitly a pooled metric over all 14 predicted steps (Eq. 20-23
define RMSE/MAE/MAPE/RAE all summed over j=1..T_out then divided by N·T_out), while the
"3/7/14 Days Ahead" columns appear (based on the metric definitions given, which are
generic over j) to also be evaluated with the same pooled formula restricted to a
shorter T_out — the paper does not fully disambiguate whether "3 Days Ahead" means
"error averaged over days 1-3" or "error at day 3 only"; given the curriculum-learning
setup (models are trained incrementally up to increasing horizons) it most plausibly
means a model/evaluation truncated to T_out=3, again averaged over days 1-3 — i.e.
**this, like STAN, is a pooled/windowed multi-step metric, not a single-lead metric**,
a second clear example of the family-wide "horizon means different things in different
papers" problem the task brief flags.

Runs: **"we perform 5 trials for each model and calculate the mean and 95% confidence
interval of results. The used random seeds are 0, 1, 2, 3, 4."** Mean±something (Table I
shows "±" values, format ambiguous whether SD or CI half-width — text says "95%
confidence interval" but the ± notation in the table is not explicitly re-labeled) is
reported for every model, including every baseline. This is the most systematically
multi-seed-reported table of the five papers (Cola-GNN reports SD via figures only, not
in its main results tables; EpiGNN reports no variance at all).

#### E. Baselines
11 baselines across three explicit classes (paper's own framing): **Mechanistic** — SIR,
SIR(Copy) [previous-week β,γ reused], MetaSIR, MetaSIR(Copy); **Spatio-temporal deep
learning** — STGCN, DCRNN, GraphWaveNet, MTGNN, AGCRN; **GNN-based epidemic models** —
CovidGNN, **ColaGNN**. All are re-run by the authors on their own Japan mobility dataset
(there is no possible number-copying from the Cola-GNN/EpiGNN ILI papers here, since the
domain/data is entirely different) — confirmed re-run given the mean±CI format applied
uniformly to every row including baselines.

#### F. Headline numbers (verified directly from Table I of the PDF)
3-Days-Ahead RMSE: MepoGNN(Adp) 141.0±7.2, MepoGNN(Dyn) **135.9±17.8** (best) vs best
non-MepoGNN baseline GraphWaveNet 223.8±46.6, ColaGNN 221.7±40.7.
7-Days-Ahead RMSE: MepoGNN(Adp) 174.6±10.1, MepoGNN(Dyn) **160.6±4.5** vs GraphWaveNet
259.9±52.2, ColaGNN 300.6±61.2.
14-Days-Ahead RMSE: MepoGNN(Adp) **261.1±16.0**, MepoGNN(Dyn) 253.2±7.5 vs GraphWaveNet
389.8±20.8, ColaGNN 388.3±23.2.
Overall RMSE: MepoGNN(Adp) 196.2±11.3, MepoGNN(Dyn) **186.1±5.0** vs GraphWaveNet
294.7±40.9, ColaGNN 310.7±31.4.
ColaGNN is consistently among the *weaker* baselines here (worse than GraphWaveNet,
AGCRN, DCRNN at most horizons) — a marked contrast to its role as the strong SOTA
reference point in the Cola-GNN-family ILI papers; on this COVID/mobility task, plain
adaptive-graph traffic-forecasting architectures (GraphWaveNet, MTGNN) beat it.

#### G. Ablations / interpretability
Table II: w/o glm (no graph-learning module — replaced presumably by the fixed
input graph), w/o propagation (metapopulation SIR module reduced to plain, non-coupled
SIR), w/o SIR (metapopulation-SIR module removed entirely). **Removing the SIR module
causes by far the largest performance drop** (e.g. Dynamic-graph Overall RMSE
186.07→290.78, +56%), which the paper explicitly attributes to the SIR module's ability
to "handle the unprecedented surge of cases" (the fifth wave in the test set) —
i.e. the mechanistic inductive bias, not the graph-learning module, is doing most of the
work; w/o glm and w/o propagation cause much smaller drops (186→194-200 range).
Interpretability case studies (Figs. 13-15): plots the learned pseudo-effective
reproduction number R̂^t over time against real policy events (state of emergency,
Olympics) and shows qualitative alignment; visualizes the learned adaptive mobility
graph against the static commuter-survey graph it was initialized from, showing the
learned graph mostly "keeps the major structure of the commuter graph" with some
deviations. A **named Limitation section** (§VIII, the only other paper besides STAN to
have one) honestly reports a failure case: the model cannot predict the sudden onset of
the sixth epidemic wave in an extra out-of-sample test period ("the number of confirmed
cases surged to thousands from continued near zero in a very short period of time... it
fails to produce accurate predictions at the beginning of this epidemic wave").

#### H. Writing
Sections: Abstract, Intro (with itemized contributions list), Related Work (2
subsections: epidemic forecasting models split mechanistic/deep-learning, SIR/
metapopulation-SIR background), Problem Formulation, Methodology (3 subsections:
metapopulation-SIR module, spatio-temporal module, graph-learning module), Data (4
subsections: epidemic, external, mobility-flow data — with real effort spent explaining
GPS-sampling-bias correction), Experiments (setting, evaluation w/ baseline
descriptions, case study, mobility-generation test), Conclusion, **Limitation** (named
section), Acknowledgments, References. 13 pages. Heavy use of Japan choropleth maps and
time-series small multiples (case counts, movement-range data, dynamic OD flow before/
after normalization) as figures — a strong template for a data-provenance section.
Explicit "why propose two types of graph learning module" subsection directly addressing
a design question a reviewer would ask — good practice. Positions itself against pure
black-box GNN epidemic models (Cola-GNN, CovidGNN — "simply treat epidemic forecasting
as ... a pure black-box manner") and against classical metapopulation-SIR calibration
methods, explicitly claiming to be the first to combine the two.

#### I. Red flags
- The "±" values in Table I are described in text as "95% confidence interval" but the
  table itself never re-states the CI half-width formula or confirms it isn't simply SD
  — genuinely ambiguous, and matters a lot for any significance claim built on this
  table.
- "3/7/14 Days Ahead" columns are not explicitly defined as single-lead vs windowed-
  average — see D above; given the metric equations provided (Eq. 20-23, generic sum
  over j=1..T_out) the most literal reading is that they are windowed averages over
  1..h, exactly the "pooled over leads 1..h" pattern the task brief specifically warned
  about as a known scoring-bug shape in a Cola-GNN fork. I could not find a sentence
  that resolves this ambiguity either way despite reading the whole paper.
- ColaGNN's poor showing here (worst or near-worst baseline) versus its strong showing in
  its own paper and EpiGNN's paper is a useful illustration that "SOTA" is highly
  dataset/domain-dependent within this same nominal "epidemic GNN" family — not a flaw
  in MepoGNN's own experiment, but a caution against treating any one paper's baseline
  ranking as transferable.
- Ablation shows the graph-learning module contributes comparatively little versus the
  SIR-coupling — somewhat undercuts the paper's own emphasis (2 of 5 contribution
  bullets are about the graph-learning module) relative to what actually drives
  performance (the mechanistic SIR module, per the ablation's own numbers).

#### J. Relevance to the new paper
Most directly relevant precedent for claim (iii) (scale-equivariant, graph-free
common-factor model with seasonal memory): MepoGNN's own ablation shows the *mechanistic/
epidemiological* component (metapopulation SIR) contributes far more than the *learned
graph* component to its performance gains, and its Limitation section honestly documents
failure on genuinely novel (unprecedented-surge) dynamics — both are strong, citable,
family-internal precedents that graph learning may be doing less work than architecture
papers claim, and that mechanistic structure (which a common-factor/seasonal-memory model
also supplies, without a learned graph at all) can be the more load-bearing component.
Its named Limitation section and honest failure case are exactly the template a
protocol-correction paper should point to as the exception, not the rule, in this
literature.

---

### Code-repo verification (Cola-GNN & EpiGNN)

Pulled `src/data.py` from both `amy-deng/colagnn` (master) and `Xiefeng69/EpiGNN` (main)
directly via `gh api`/raw.githubusercontent.com (not a paraphrase of a summarizer for the
key lines — the relevant snippets were quoted back verbatim by the fetch and cross-
checked against the papers' own text above).

**Split** (identical in both repos):
```python
self.train_set = train_set = range(self.P+self.h-1, train)
self.valid_set = valid_set = range(train, valid)
self.test_set  = test_set  = range(valid, self.n)
```
with `train = int(args.train * self.n)`, `valid = int((args.train + args.val) * self.n)`
— i.e. **chronological**, index-ordered, fractions from CLI args (both papers' defaults
match their stated 50/20/30).

**Normalization** (identical in both repos):
```python
self.max = np.max(train_mx, 0)
self.min = np.min(train_mx, 0)
self.dat = (self.rawdat - self.min) / (self.max - self.min + 1e-12)
```
where `train_mx` is built only from the training-split batches — **fit on train only**,
then applied to the full series. This is leakage-safe and is **not** explicitly stated in
either paper's prose (both papers only say "data is normalized to 0-1 range... maximum
value... set to 1" without saying train-only) — a reader relying on the papers alone
could not confirm this without the code.

**Horizon → target construction** (identical in both repos):
```python
end = idx_set[i] - self.h + 1
start = end - self.P
Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])
```
`Y` is read at a **single index** `idx_set[i]`, with the input window `[start, end)`
ending exactly `h` steps before it. **This confirms, at the code level, that both
Cola-GNN's and EpiGNN's official evaluation is single-lead (predict exactly step t+h),
never pooled/averaged over a range of leads t+1..t+h or t+h..t+2h-1.** The scoring-bug
shape the task brief warned about (pooled leads h..2h-1 in a *fork* of Cola-GNN) is **not
present in the original amy-deng/colagnn repo or in Xiefeng69/EpiGNN** — both are single-
lead as designed and as described in their papers. (I did not have access to inspect the
specific fork mentioned in the brief, since none was named; this finding is about the
two canonical upstream repos only.)

`amy-deng/colagnn/src/train.py` additionally confirms RMSE/PCC are computed **per-state
then averaged** (`pcc_tmp.append(pearsonr(...)[0]); pcc_states = np.mean(pcc_tmp)`), and
that denormalization back to real counts happens before computing the reported metric
(`y_true_states = y_true_mx.numpy() * (max-min) + min`) — consistent with the papers'
stated "RMSE/PCC computed after projecting normalized values into the real range."

The fact that EpiGNN's `data.py` is essentially line-for-line identical to Cola-GNN's
(same variable names `self.h`, `self.P`, `idx_set`, same slicing logic) is itself
evidence that EpiGNN forked Cola-GNN's data pipeline wholesale rather than
reimplementing it — consistent with, and probably the underlying cause of, the copied
(and mismatched) Cola-GNN baseline numbers noted in §2.F/I above.

---

### Cross-paper synthesis (~300 words)

Despite sharing a common lineage — Cola-GNN → EpiGNN directly forks Cola-GNN's data
pipeline; STAN and MepoGNN cite and re-run Cola-GNN as a baseline — "horizon" means at
least three different things across this family. Cola-GNN and EpiGNN train one model per
single lead h and evaluate at that lead only (verified in both papers' text and,
independently, in both repos' `data.py`, which are near-identical). STAN and MepoGNN
instead predict an entire L_P/T_out-day window in one shot and report metrics that read,
at a glance, like single-horizon numbers ("L_P=15", "7 Days Ahead") but are actually
pooled/averaged over the whole window — MepoGNN's own equations (generic sums over
j=1..T_out) leave this genuinely ambiguous even on a full careful read. This is exactly
the family of scoring ambiguity the task brief flagged, present independently in two of
the five papers, without needing to invoke any specific buggy fork.

"SOTA" is also domain-contingent in a way the family rarely acknowledges: ColaGNN is the
strongest non-proposed baseline in its own paper and in EpiGNN's (both on weekly ILI
data), but one of the weakest baselines in MepoGNN's daily-COVID/mobility setting, where
generic adaptive-graph traffic models (GraphWaveNet, MTGNN) win instead — a caution
against treating any single paper's baseline ranking as portable. Baseline-number
provenance is also imperfectly tracked: EpiGNN marks its Cola-GNN row with an asterisk
for "reported in the corresponding reference," yet the marked numbers don't match
Cola-GNN's own paper (verified by reading both tables directly), and EpiGNN's CNNRNN-Res
numbers are reused verbatim from Cola-GNN's table without any such marker at all.

On writing: only STAN and MepoGNN have a named Limitations section and only STAN reports
formal significance tests; Cola-GNN and EpiGNN report averages over 10/5 seeds but never
pair them with a hypothesis test. All five position novelty primarily via ablation-delta
against a graph-free variant, never via direct inspection of what the learned attention/
graph converged to — the gap the new paper's attention-collapse claim (ii) is built to
fill.

---

## G2: Architectural successors on the Cola-GNN benchmark family (2023-2026)

Access method note: all papers read via WebFetch of arXiv HTML (`arxiv.org/html/<id>`), PLOS/PMC full text, or abstract pages, plus WebSearch for papers without open full text. WebFetch summarizes through an intermediate small model, so exact numbers below were obtained by a *second, targeted* fetch asking for verbatim table quotes wherever a headline number mattered (HeatGNN Table III, EARTH Table 1). Where I could not get primary full text (STTGNN, EASTG, CSTGNN) this is stated explicitly, not glossed over.

---

### 1. HeatGNN

**A. Bibliographic.** Yufan Zheng, Wei Jiang, Tong Chen, Alexander Zhou, Nguyen Quoc Viet Hung, Choujun Zhan, Hongzhi Yin. arXiv:2411.17372 (submitted 26 Nov 2024; v2 fetched). No venue is stated in the arXiv metadata — appears to still be a preprint, not a confirmed peer-reviewed venue. Code: "anonymous.4open.science/r/HeatGNN-14DB" (anonymized review repo, not a permanent citable release).

**B. Claimed contributions (abstract, verbatim excerpt).** "Recent studies... bear an over-simplified assumption that two locations... with similar observed features in previous time steps will develop similar infection numbers in the future. In fact... there exists strong heterogeneity of its intrinsic evolution mechanisms across geolocation and time... To address this challenge, we propose a Heterogeneous Epidemic-Aware Transmission Graph Neural Network (HeatGNN)... By binding the epidemiology mechanistic model into a GNN, HeatGNN learns epidemiology-informed location embeddings... Experiments on four benchmark datasets have revealed that HeatGNN outperforms various strong baselines. Moreover, our efficiency analysis verifies the real-world practicality of HeatGNN on datasets of different sizes." (Note: an earlier automated read of the same URL mis-transcribed this as "three benchmark datasets" — the verbatim quote confirms **four**.)

**C. Method (5-8 lines).** EpiGNN-style spatio-temporal graph learning (STGL) backbone extracts multi-scale temporal features and a transmission-risk encoding. A separate epidemiology-informed embedding learner (EIEL) uses five MLPs to predict per-location time-varying SIR variables (S, I, R, β, γ) from the ST embeddings. These SIR-parameter embeddings are used to build a time-varying "mechanistic affinity graph" via cosine similarity + sparsification (i.e., a second, physics-flavored dynamic graph alongside the geographic/attention graph). A GCN propagates over this transmission graph; its output is concatenated with the ST embedding and decoded. Loss = forecasting loss + a physics-informed regularizer tying the predicted SIR trajectory to observed cases.

**D. Data & protocol (quoted).** Four datasets: Japan-Prefectures (47×348, Aug 2012–Mar 2019, ILI), US-Regions (10×785, 2002–2017, ILI), US-States (49×360, 2010–2017, ILI), Australia-COVID (8×556, Jan–Aug 2020, daily confirmed). Split: "we divide the dataset into training, validation, and test sets in chronological order with a ratio of 60%-20%-20%." Window w=20; horizons h∈{2,5,7,12} — **note this horizon grid (2,5,7,12) differs from both Cola-GNN's own grid (2,3,5,10,15) and EpiGNN's (3,5,10,15)**, so no single h is numerically comparable to either source paper's table without recomputation. Normalized on training statistics. "All experimental results are the average of 5 randomized trials" — **no standard deviations or CIs reported anywhere in the tables**. Adam, lr=1e-3, wd=5e-4, batch 32, early stopping patience 200, up to 1500 epochs. Baselines include SIR/AR/ARMA/VAR/GAR, LSTM/GRU/RNN-Attn, LSTNet/CNNRNN-Res, STGCN/MGNN/TMGNN/Cola-GNN/EpiGNN, and "Epi-Cola-GNN." No explicit persistence/naive baseline beyond SIR and AR-family. Metrics: RMSE, PCC only (no MAE, no CRPS/calibration).

**E. Baselines.** ~17 methods listed above, including Cola-GNN and EpiGNN — re-run, not copied verbatim (see F for the mismatch vs. Cola-GNN's self-reported numbers).

**F. Headline numbers (Table III, RMSE ×10³ and PCC, verbatim from targeted re-fetch).**

| Dataset | h | HeatGNN RMSE / PCC | EpiGNN RMSE / PCC | Cola-GNN RMSE / PCC |
|---|---|---|---|---|
| Japan-Prefectures | 2 | 1.149 / 0.917 | — | — |
| Japan-Prefectures | 5 | 1.378 / 0.884 | 1.448 / 0.869 | 1.573 / 0.834 |
| Japan-Prefectures | 7 | 1.735 / 0.780 | — | — |
| Japan-Prefectures | 12 | 1.685 / 0.773 | — | — |
| US-Regions | 2/5/7/12 | 0.541 / 0.941, 0.852 / 0.866, 0.922 / 0.856, 1.024 / 0.824 | — | — |
| US-States | 2 | 0.142 / 0.953 | 0.150 / 0.952 | 0.148 / 0.953 |
| US-States | 5/7/12 | 0.186 / 0.921, 0.200 / 0.911, 0.240 / 0.870 | — | — |
| Australia-COVID | 2/5/7/12 | 0.315/0.993, 0.334/0.993, 0.411/0.991, 0.466/0.986 | 0.381/0.993 (h=5) | 0.406/0.994 (h=5) |

**Copied-vs-re-run check:** Cola-GNN's own paper reports Japan RMSE at h=5 = **1051** (task reference set). HeatGNN's re-run of Cola-GNN at Japan h=5 = **1573** — 50% higher than Cola-GNN's self-reported number. This is a re-run under a different (60/20/20, w=20, h=2/5/7/12) protocol, not a copy, and the re-run makes Cola-GNN look substantially worse than in its own paper, which inflates HeatGNN's apparent margin.

**G. Improvement magnitudes.** Paper does not state a single headline "% improvement" number in the fetched sections; gains vary 5-15% RMSE depending on dataset/horizon from the table above. Averaged over 5 seeds, no seed count/test-set-size-based test of significance is offered, so whether the improvement exceeds noise cannot be assessed from the paper itself.

**H. Ablations.** Table IV: "w/o PL" (no physics loss), "w/o TG" (no transmission graph), "w/o TG+EIEL" (remove both) — all degrade performance, TG module's removal "more pronounced." Table VIII: robustness under injected 10-30% Gaussian noise and missing nodes — HeatGNN degrades less than Cola-GNN/EpiGNN. No seeds/CI on ablation deltas either.

**I. Writing.** Standard IEEE/ACM-style ML paper: intro (heterogeneity motivation) → related work → method (STGL/EIEL/TG) → experiments (main table, ablation, robustness, efficiency) → conclusion. No explicit "Limitations" section found in the fetched content. Reproducibility: anonymized code link only (no permanent DOI/archive), hyperparameters given in text.

**J. Red flags.** (1) Horizon grid incompatible with source papers' own grids, defeating direct comparison while table layout visually invites it. (2) Re-run Cola-GNN numbers are far worse than Cola-GNN's self-reported numbers at the same nominal horizon (1573 vs 1051 at "h=5"), which — combined with the different window/split — means the margin over Cola-GNN is partly a protocol artifact, not (only) an architecture improvement. (3) No variance/significance reporting despite averaging 5 trials. (4) Code link is a non-archival "anonymous.4open.science" URL even in a non-anonymous arXiv posting.

**K. Relevance to the new paper.** Directly useful as a worked example of (i) horizon-grid incomparability across "the same" benchmark name, (ii) a re-run baseline that is quietly worse than the source paper's own number, inflating claimed gains, and (iii) 5-trial averaging with no variance/CI — exactly what a power-analysis-and-calibration paper should contrast itself against. HeatGNN's "mechanistic heterogeneity" graph is a second attention/graph module layered on EpiGNN — a good foil for the argument that added graph machinery keeps not paying for itself once evaluated fairly.

---

### 2. EARTH

**A. Bibliographic.** Guancheng Wan, Zewen Liu, Max S.Y. Lau, B. Aditya Prakash, Wei Jin. "Epidemiology-Aware Neural ODE with Continuous Disease Transmission Graph." arXiv:2410.00049 (v1 28 Sep 2024, v2 10 Nov 2024). Task states ICML 2025 — not visible in the arXiv metadata itself (arXiv shows no journal-ref field in what was fetched), so the ICML venue claim is taken from the task brief, not independently confirmed from the arXiv page. Code: "will be available at https://github.com/Emory-Melody/EpiLearn" (an EpiLearn library repo, i.e., not a dedicated single-paper repo — could not verify code is actually present without a separate fetch).

**B. Claimed contributions.** "We introduce an innovative end-to-end framework called Epidemiology-Aware Neural ODE with Continuous Disease Transmission Graph (EARTH)" — first framework to harmonize neural ODEs with epidemic mechanisms; models global epidemic trends guiding local regional transmission; cross-attention fusion of global and local signals.

**C. Method (5-8 lines).** Epidemic-Aware Neural ODE (EANO) treats S/I/R as latent ODE states per node, with the transmission and recovery terms parameterized by learned weight matrices operating on a graph-aggregated infectious term (Σ e_vu I_u(t)) — i.e., a continuous-time GNN-SIR hybrid. A Global-guided Local Transmission Graph (GLTG) module computes cross-region similarity via Dynamic Time Warping on global infection-trend features and fuses it with a static geographic adjacency to form a dynamic graph. Cross-attention fuses the global trend embedding with the local ODE state for the final forecast head.

**D. Data & protocol (quoted).** Three datasets only: Australia-COVID, US-Regions, US-States — **no Japan-Prefectures**. Window T=20; horizons h=5,10,15 (matches the standard grid subset used by Cola-GNN/EpiGNN, unlike HeatGNN). Metrics: RMSE (ℛ) and a Peak Time Error 𝒫 (MAE on significant peaks) — a metric not used by Cola-GNN/EpiGNN's own papers, so peak-error comparisons to those source papers are not literally possible. "We repeat each experiment five times for each dataset and record the average results" — again **no std/CI reported** in the main table. SGD (momentum 0.9), lr=1e-3, hidden dim 64. Baselines: 14 methods incl. VAR, LSTM, DCRNN, STGCN, EpiGNN, ColaGNN, EpiColaGNN. Train/val/test split fractions and whether chronological: **not stated** in the fetched sections (both fetch passes came back empty on this point) — this is a genuine gap in what I could verify, not an assumption.

**E. Baselines.** Re-run (see F: numbers diverge sharply from Cola-GNN's and EpiGNN's own papers).

**F. Headline numbers (Table 1, RMSE, verbatim from targeted re-fetch).**

| Dataset | h | EARTH | EpiColaGNN | ColaGNN | EpiGNN |
|---|---|---|---|---|---|
| Australia-COVID | 5 | 156.8 | 204.3 | 224.2 | 210.3 |
| Australia-COVID | 10 | 177.6 | 345.4 | 544.8 | — |
| Australia-COVID | 15 | 225.3 | 886.0 (range max) | 795.8 (range max) | 764.2 (range max) |
| US-Regions | 5/10/15 | 1080 / 1244 / 1301 | 1185–1371 (range) | 1148–1552 (range) | 1136–1444 (range) |
| US-States | 5 | 243.2 | 286.1 | **299.1** | 288.5 |
| US-States | 10/15 | 277.8 / 300.1 | up to 375.1 | up to 339.4 | up to 391.6 |

**Copied-vs-re-run verification (the task's specific check):** Cola-GNN's own paper reports US-States RMSE at h=5 = **202**. EARTH's re-run lists ColaGNN US-States h=5 = **299.1** — confirmed as stated in the task brief, a ~48% inflation of Cola-GNN's own self-reported error at the nominal same horizon. This is unambiguous evidence of a re-run under a different protocol (not a typo-level copy error), and it substantially widens EARTH's apparent margin over Cola-GNN. The same pattern holds for EpiGNN: EARTH's US-States figures for EpiGNN (288.5 at h=5) vs. EpiGNN's own reported US-States numbers (160/186/220/236 at h=3/5/10/15) — EARTH's re-run EpiGNN is also much worse than EpiGNN's self-reported numbers.

**G. Improvement magnitudes.** On Australia-COVID h=10, EARTH (177.6) vs best listed baseline range (EpiColaGNN 345.4) is roughly a 48% RMSE reduction — a very large claimed margin, but built on baselines whose re-run numbers are already 1.5-3x worse than their own papers' self-reported numbers, so the "48%" is not a clean architecture effect. No test-set-size/seed-based significance test is offered anywhere (5 repeats, means only).

**H. Ablations (Table 2, Australia-COVID h=5, RMSE).** "w/o Both" (no EANO, no GLTG) = 267.4; "w EANO" (only EANO) = 178.6; "w GLTG" (only GLTG) = 232.4; full EARTH = 156.8. Monotonic, consistent with a real contribution from both parts, though again without variance bars.

**I. Writing.** No dedicated "Limitations" section found. Otherwise standard: intro, related work, EANO/GLTG method sections, experiments (main table + ablation + robustness to irregular sampling + horizon sweep 1-20), conclusion.

**J. Red flags.** (1) The Cola-GNN/EpiGNN re-run numbers are dramatically worse than those papers' own self-reported numbers at nominally the same horizon and dataset (US-States h=5: 299.1 and 288.5 vs. self-reported 202 and 160) — this is the clearest instance found in this batch of a paper's re-run baselines being quietly weaker than the source papers, inflating the proposed model's apparent lead. (2) No Japan-Prefectures dataset despite being part of the "standard four." (3) A bespoke Peak Time Error metric not present in Cola-GNN/EpiGNN, making that axis of comparison paper-specific by construction. (4) Train/val/test split description not found in the accessible text — a reproducibility gap.

**K. Relevance to the new paper.** This is the single strongest documented example in this batch of the exact protocol-inflation problem your new paper is meant to correct — worth citing/quoting the 299.1-vs-202 and 288.5-vs-160 gaps directly as evidence that re-run baselines in this literature are systematically weakened, not just occasionally.

---

### 3. EpiHybridGNN

**A. Bibliographic.** Xiangxin Kong, Hang Wang, Yutong Li, Yanghao Chen, Zudi Lu. arXiv:2511.15469 (submitted 19 Nov 2025), listed under stat.CO — this reads as a methods/applied-stats paper rather than an ML-venue submission; no separate venue given. No code repository link found in the fetched text ("Nvidia 2060 GPU," "PyTorch 2.2.1" reported, but no GitHub URL).

**B. Claimed contributions (abstract, verbatim excerpt).** "Building on in-depth review and assessment of two popular graph neural network (GNN)-based regional epidemic forecasting models of EpiGNN and ColaGNN, we propose a novel hybrid graph neural network model, EpiHybridGNN, which integrates the strengths of both... Our EpiHybridGNN is therefore designed to combine the advantages of both EpiGNN, in its risk encoding and RAGL, and ColaGNN, in its long-term forecasting capabilities and dynamic attention mechanisms."

**C. Method (5-8 lines).** Multi-scale dilated temporal convolutions (short + long-term, ColaGNN-style) feed a transmission-risk encoding module (EpiGNN-style local geographic-degree risk + global attention-based risk). A dynamic graph fuses RNN-based cross-location attention with a geographic prior and any external relational data. A multi-layer GCN with layer norm and residual connections propagates over this fused graph; the decoder concatenates GCN output with RNN hidden state plus an optional residual/skip window (ColaGNN's AR-residual trick).

**D. Data & protocol (quoted).** Four datasets: Australia-COVID (8×556 daily), Japan-Prefectures (47×348 weekly), US-Regions (10×785 weekly), US-States (49×360 weekly) — the standard four. Split: "chronologically divided into training (50%), validation (20%), and testing (30%)" — **this differs from the 60/20/20 used by HeatGNN and (per the ledger) by Cola-GNN/EpiGNN's own protocol**, another instance of nominally-same-dataset, different-split incomparability. Window T=20; an unusually long horizon list h={2,5,8,11,14,17,20,23,26,29,32} — far beyond the 2-15 range used elsewhere, testing forecasting out to 32 steps. Batch 128, lr 1e-3, dropout 0.2, Adam wd 5e-4. Metrics: RMSE, MAE, PCC. Baselines: EpiGNN, ColaGNN, STGCN, GAR, VAR, LSTM, CNNRNN-Res — no explicit persistence/naive baseline stated beyond GAR/VAR.

**E. Baselines.** EpiGNN and ColaGNN re-run (own implementation, not copied — see F).

**F. Headline numbers (Australia-COVID, h=2, verbatim).**

| Model | MAE | RMSE | PCC |
|---|---|---|---|
| EpiGNN | 127.25 | 388.50 | 0.9942 |
| ColaGNN | 38.54 | 180.31 | 0.9974 |
| EpiHybridGNN (Hybrid) | 23.40 | 116.44 | 0.9988 |

No Japan-Prefectures/US-Regions/US-States numeric table was returned by the fetch beyond this Australia-COVID excerpt; the source content likely has full four-dataset tables but the automated read did not surface them verbatim, so I am not reporting numbers I did not actually see. This is a genuine coverage gap in my read, not a claim that the data don't exist.

**G. Improvement magnitudes.** At Australia-COVID h=2, EpiHybridGNN's RMSE (116.44) is ~35% lower than ColaGNN's re-run RMSE (180.31) and ~70% lower than EpiGNN's re-run RMSE (388.50). No seed count given beyond an implicit single run (the fetched text did not mention multi-seed averaging), so this cannot be assessed for significance from what's stated.

**H. Ablations.** Paper has ablation (6.4.2), sensitivity analysis (6.4.3), interpretability analysis (6.4.4) sections per the fetch, but no numeric values were extracted — another explicit read gap.

**I. Writing.** Framed as a comparative/hybrid-engineering paper ("in-depth review and assessment" of two prior models) rather than a from-scratch novel architecture — the contribution is explicitly a combination/ablation exercise. Limitations acknowledged directly in text: "ColaGNN's underlying graph topology fundamentally relies on pre-defined connections" and EpiGNN "shows decreased long-term prediction performance" as horizon increases — refreshingly candid framing of prior weaknesses, though it is describing the baselines' limitations rather than its own model's.

**J. Red flags.** (1) 50/20/30 split diverges from the 60/20/20 used elsewhere in this literature and from Cola-GNN/EpiGNN's own reported protocol, so its reported numbers are not directly comparable to either source paper even before considering re-run variance. (2) No code link provided. (3) stat.CO arXiv category and single-GPU (2060) footprint suggest a smaller-scale/less rigorously reviewed effort than the ICML/DMKD/KBS entries in this group. (4) I could not verify multi-seed/variance reporting from the accessible text — flagged as unconfirmed rather than assumed absent.

**K. Relevance.** A clean example of "recombination" papers in this space (splice EpiGNN + ColaGNN components) whose main lever for beating both source papers is, again, a different split fraction (50/20/30 vs. 60/20/20) — reinforces the argument that apples-to-apples comparison requires literally re-running everything under one fixed protocol, which is exactly what a corrected-evaluation paper should demonstrate explicitly.

---

### 4. STTGNN

**A. Bibliographic.** "A multi-scale spatio-temporal transformer with region-aware graph learning for epidemic forecasting," Knowledge-Based Systems, 2026 (per WebSearch snippet, "published... April 2026"), https://www.sciencedirect.com/science/article/abs/pii/S0950705126006374. **Full text is paywalled** — WebFetch on the ScienceDirect abstract URL returned HTTP 403 Forbidden. No arXiv preprint, author-copy, or ResearchGate mirror was found via WebSearch (search for title + "arxiv" / "preprint" surfaced only the same ScienceDirect abstract-page listing and unrelated papers). I could not read the full paper; what follows is limited strictly to what is visible in the ScienceDirect abstract/search-engine snippet.

**B. Claimed contributions (from indexed abstract/snippet only).** STTGNN "integrates multi-scale temporal convolutions, a two-stage spatio-temporal Transformer, and a region-aware graph learner enhanced with degree-aware gating." Bullet-style claims recovered from the snippet: spatial dependencies are modeled before temporal ones via a two-stage Transformer ("reflecting the inductive bias that inter-regional influence patterns precede their temporal propagation"); a region-aware graph learner infers directed, time-varying adjacency, adaptively fused with static geography via degree-aware gating "to enhance model robustness under low-incidence and noisy scenarios"; "Experiments on Japan-Prefectures and US-Regions datasets demonstrate that STTGNN consistently outperforms statistical baselines, deep sequence models, and state-of-the-art spatio-temporal graph methods in terms of RMSE, PCC, and peak error."

**C. Method (from snippet only, not full text).** Multi-scale temporal decomposition into short/medium/long-term components, each processed by a temporal Transformer; a two-stage Transformer backbone sequences spatial-then-temporal attention; a region-aware graph learner produces directed time-varying adjacency; degree-aware gating fuses this with static geographic adjacency. This is a reconstruction from the abstract snippet, not a verified read of the method section, equations, or figures.

**D-K.** **Not assessable — full text inaccessible.** Only two datasets are named in the snippet (Japan-Prefectures, US-Regions), notably omitting US-States and Australia-COVID, but I cannot confirm this is the complete dataset list without the full text. No split fractions, horizons, seeds, baselines, or numeric results were recoverable. No headline numbers can be reported without fabricating them, so none are given here.

**J. Red flag (procedural, not content-based).** This is the second KBS/Elsevier-family paper in the group (with BDGSTN) — Elsevier epidemic-forecasting papers in this space are consistently paywalled with no preprint culture, which itself is worth noting as a field-level access/reproducibility problem distinct from any specific paper's rigor.

---

### 5. PISID

**A. Bibliographic.** Satoki Fujita, Tatsuya Akutsu (Bioinformatics Center, Kyoto University). "Enhancing epidemic forecasting with a physics-informed spatial identity neural network." PLOS ONE 20(9): e0331611, published 15 Sep 2025. Full text read via PMC (PMC12435659). Code: https://github.com/satoki-fujita/PISID.

**B. Claimed contributions (paraphrase of framing, method section is explicit).** The model — "Physics-Informed Spatial Identity neural network" — integrates an STID-style (Spatio-Temporal IDentity) graph-free neural encoder with an SIR module. Motivation stated directly: "deep learning-based models have increasingly leveraged graph structures to capture the spatial dynamics of epidemic spread... this approach often increases model complexity, and the resulting performance gains may not justify the added burden. In some cases, it may even lead to overfitting." This is an explicit graph-skeptical framing — the most directly relevant paper in this group to a graph-free successor argument.

**C. Method (5-8 lines).** A spatio-temporal neural network module: temporal information embedded via a fully-connected layer, spatial identity embedded via a *learnable* per-location matrix E ∈ ℝ^(M×D) (no adjacency at all — locations are just indices into a learned embedding table, STID-style), concatenated and passed through L MLP layers with residual connections to output per-location, per-timestep SIR parameters β (infection rate) and γ (recovery rate). An SIR module then rolls these parameters forward through discrete SIR dynamics to produce the case forecast; infectious counts are inferred through the SIR equations rather than observed directly (only confirmed-case data and population are required as input — no recovered-case data needed).

**D. Data & protocol (quoted).** Datasets are **not** the Cola-GNN four: Japan (47 prefectures, daily, Jan 16 2020–May 8 2023, COVID) and US (51 states, daily, Jan 22 2020–Mar 9 2023, COVID) — both COVID-era daily case series, evaluated separately by dominant-variant period (Delta vs. Omicron windows), not the ILI-era weekly Cola-GNN benchmark. Split "6:1:3 ratio" train/val/test (i.e. 60/10/30, not 60/20/20). Input history T_in ∈ {14,28} days, horizon T_out ∈ {14,28} days (short, COVID-appropriate horizons — not the 2-32 range used by the ILI-era papers). 7-day moving-average preprocessing. Normalization on training mean/std. Curriculum learning: horizon grown from 1 up to T_out during training. Adam lr=1e-3, wd=1e-8, batch 32, up to 300 epochs, early-stop patience 20. **"Five times with different random initializations" reported with mean AND std** — this is the only paper in this batch of six primary reads that reports standard deviations in its main results table. Loss = MAE.

**E. Baselines.** SIR, ARMA, GAR, RNN, DCRNN, LSTNet, STGCN, GWNet, ColaGNN, FourierGNN, STID. EpiGNN is **not** a baseline (not mentioned as a direct comparison, only possibly in related work). ColaGNN is re-run on PISID's own COVID datasets, not copied from Cola-GNN's ILI-era paper — so no direct "copied vs. re-run" check against Cola-GNN's self-reported ILI numbers is meaningful here (different disease, different years, different horizon units).

**F. Headline numbers (with std, T_in=T_out=28, verbatim).**
- Japan, 2022/01/01–2023/05/08: PISID MAE 549.221 (±61.100), RMSE 1160.686 (±123.978), MAPE 0.750 (±0.237), CCC 0.709 (±0.076); SIR baseline MAE 594.251, RMSE 1469.205.
- US, 2021/12/01–2023/03/09: PISID MAE **370.447** (±6.435), RMSE **806.841** (±26.431), MAPE **2.409** (±0.136), CCC **0.807** (±0.012); GWNet MAE 428.229 (±17.755), RMSE 928.569 (±60.528).
- Text states PISID ranked "either the best or the second-best MAE" across scenarios — an honest, non-absolute claim (doesn't claim best-everywhere).

Because these are COVID/daily datasets rather than Japan-Prefectures/US-Regions/US-States/Australia-COVID, **no direct numeric comparison to Cola-GNN's or EpiGNN's own reported RMSE values is possible** — this paper is only nominally in the "Cola-GNN benchmark family" by reusing Japan/US geography and ColaGNN as one re-implemented baseline, not by using the actual benchmark datasets or splits.

**G. Improvement magnitudes.** ~13-15% RMSE/MAE reduction vs. best listed baseline (GWNet) on US; larger margin vs. plain SIR on Japan. With 5-seed mean±std reported, this is the only paper in the batch where a reader could in principle back out an approximate z-test — though the paper itself does not run one.

**H. Ablations/interpretability.** Table 4: encoder variants (MLP+spatial-ID beats RNN, TCN, Transformer, GWNet, and MLP-without-spatial-embedding) — a direct ablation showing the *graph-free* spatial-identity embedding beats several graph/sequence encoders on this task. Figure 3: hidden-dim sweep (D=32 chosen; larger D overfits). Interpretability: effective reproduction number R_e recovered from the fitted SIR parameters and shown to track real-world policy interventions (lockdowns, reopenings) — a genuine mechanistic-interpretability check rather than an attention-map dressing.

**I. Writing.** Has an explicit **Limitations** section (rare in this batch): (1) relies solely on confirmed-case input, sensitive to external unobserved factors; (2) struggles with sudden non-stationary trend shifts; (3) evaluation limited to COVID-19; (4) SIR-applicability assumption may not transfer to diseases needing SIS or other compartment structures. Efficiency reported concretely: ~27K parameters, 0.45 s/epoch vs. GWNet 5.4 s/epoch and ColaGNN 2.76 s/epoch. Code publicly released under the author's own GitHub, not an anonymized link.

**J. Red flags.** Relatively few for this batch: (1) dataset naming ("Japan," "US") invites confusion with the Cola-GNN "Japan-Prefectures"/"US-States" benchmarks despite being entirely different (COVID daily vs. ILI weekly, 2020-2023 vs. 2002-2019) — a reader skimming only dataset names could wrongly assume comparability. (2) ColaGNN is the only Cola-GNN-family baseline re-run; EpiGNN is absent, so the "beats EpiGNN" claim cannot be made or checked at all here.

**K. Relevance to the new paper — high.** This is the closest existing analogue in the group to "graph-free, scale-equivariant, common-factor-with-seasonal-memory" positioning: PISID explicitly argues learned graph structure adds complexity/overfitting risk without matching benefit, replaces it with a learned per-location embedding table (no adjacency), and grounds the network in a mechanistic SIR structure with physically interpretable output (R_e). Its ablation (Table 4) is direct evidence that a graph-free spatial-identity encoder matches or beats graph/attention encoders including GWNet and a ColaGNN-style baseline. The 5-seed mean±std reporting is also the best-practice example in this batch to cite when arguing for proper uncertainty quantification. Main difference from your planned paper: PISID's mechanistic core is SIR-parameter regression rather than a renewal/common-factor decomposition, and it evaluates on COVID daily series rather than the ILI-era Cola-GNN four, so it cannot itself serve as an apples-to-apples baseline for a corrected-protocol re-evaluation — but it is strong prior art for the "graph-free matches or beats graph-based" argument.

---

### 6. BDSTGNN (paper self-identifies as BDGSTN)

**A. Bibliographic.** Junkai Mao, Yuexing Han, Gouhei Tanaka, Bing Wang. arXiv:2312.00485 (submitted 1 Dec 2023); task brief lists venue as KBS 2024 (not independently re-verified from the arXiv page itself, which shows no journal-ref in what was fetched). **Important naming note:** the paper's own abstract and body consistently call the model **BDGSTN** ("Backbone-based Dynamic Graph Spatio-Temporal Network"), not "BDSTGNN." The user's local clone is named `BDSTGNN` and its main class file is `model/BDSTGNN.py`, but this appears to be the cloner's/task's shorthand rather than the paper's own acronym — worth flagging so it isn't mistaken for a different paper. Code: local mirror inspected directly at `C:\Users\ajaoo\Documents\GitHub\BDSTGNN` (README confirms same title/authors: "BDSTGNN — Backbone-based Dynamic Spatio-Temporal Graph Neural Network for Epidemic Forecasting").

**B. Claimed contributions (abstract, verbatim excerpt).** "Many deep learning-based models focus only on static or dynamic graphs when constructing spatial information, ignoring their relationship. Additionally, these models often rely on recurrent structures, which can lead to error accumulation and computational time consumption... we propose... (BDGSTN). Intuitively, the continuous and smooth changes in graph structure[s] make adjacent graph structures share a basic pattern. To capture this property, we use adaptive methods to generate static backbone graphs... and temporal models to generate dynamic temporal graphs..., fusing them... To overcome potential limitations associated with recurrent structures, we introduce a linear model DLinear... Finally, we... measure the significance of backbone and temporal graphs by using information metrics..."

**C. Method (5-8 lines).** A "backbone" static graph is learned via trainable node embeddings (A_back = EEᵀ, a low-rank learned adjacency, no recurrence). A "temporal" graph is derived from TCN features per timestep (A_temp = H_TCN H_TCNᵀ). The two are fused, softmax-normalized, into a single dynamic adjacency per timestep (A_dyn). Temporal dependence is modeled by DLinear (trend/seasonal decomposition + linear layers, i.e., explicitly non-recurrent) rather than an RNN/attention temporal module. A GCN aggregates over A_dyn; an auxiliary epidemiology module fits SIR parameters and a joint MAE loss combines neural and SIR-based forecasts.

**D. Data & protocol (quoted/inspected).** Two datasets, both COVID (not the ILI Cola-GNN four): US state-level (52×245, 2020-05-01 to 2020-12-31) and Japan prefecture-level (47×151, 2022-01-15 to 2022-06-14). Split: paper text says 60/20/20 chronological; **the local run script (`main.py`) instead hard-codes absolute window counts** `test_window=50`, `valid_window=50` with `history_window=5`, `pred_window=5`, `slide_step=5`, and `seed=1234` — i.e., the actual training entry point in the repo does not literally implement a percentage split, it implements fixed-length train/val/test windows (which for the 245-day US series is roughly a 60/20/20-ish split by coincidence of series length, but is not parameterized as a fraction, and for the 151-day Japan series the same fixed windows would carve up the data differently). This is a genuine discrepancy between the paper's stated protocol and the shipped code's default script, worth flagging rather than assuming the paper's prose is what actually ran. Horizons: L=5,10 "short-term," L=15,20 "long-term" (paper); code default constants show pred_window=5 as the hard-coded default, meaning the other horizons likely require manually editing constants at the top of `main.py`, not a `--horizon` CLI flag — a reproducibility friction point (no argparse; constants edited by hand, single global `seed=1234`, no evidence of multi-seed looping in `main.py` itself). Normalization: range (0,1) (min-max, not z-score). Metrics: MAE, RMSE, MAPE, PCC, CCC.

**E. Baselines.** MPSTAN, STAN, CovidGNN, plus graph-construction ablation baselines (geography-based, gravity-based, PCC-based). **No Cola-GNN or EpiGNN appears in the results tables** — this paper does not benchmark against the Cola-GNN/EpiGNN lineage at all, despite being grouped here as an "architectural successor" on that benchmark family; it is really a COVID-era competitor to STAN/MPSTAN-style models.

**F. Headline numbers (US, L=20, verbatim).**

| Model | MAE | RMSE | MAPE | PCC | CCC |
|---|---|---|---|---|---|
| BDGSTN | 11139 | 21304 | 18.06% | 98.78% | 98.23% |
| MPSTAN | 12728 | 22923 | 18.68% | 98.81% | 97.91% |
| STAN | 18679 | 36180 | 26.81% | 96.09% | 94.52% |
| CovidGNN | 26985 | 57085 | 24.57% | 92.64% | 84.95% |

Japan, L=20: BDGSTN MAE 1313/RMSE 2897 vs. MPSTAN MAE 1854/RMSE 4014. **No Japan-Prefectures/US-Regions/US-States/Australia-COVID numbers exist in this paper at all** — cannot populate the requested cross-paper comparison table for this entry.

**G. Improvement magnitudes.** Stated/derivable: 12.48% MAE / 7.06% RMSE improvement over MPSTAN on US L=20; 29.18% MAE / 27.83% RMSE improvement on Japan L=20. No seed count stated for the main table (contrast with the hard-coded single `seed=1234` in the shipped script) and no significance test — the large Japan-dataset margin (~29%) on a comparatively short 151-day series with a fixed 50-observation test window is exactly the kind of result a power analysis should be skeptical of.

**H. Ablations.** "w/o Loss" (no epidemiological/SIR loss term): US L=20 MAE 11209 vs. full 11139 — a very small (0.6%) contribution from the physics loss, arguably within noise, yet framed as validating the module. "w/o Trend" (no DLinear trend decomposition): MAE 12359 — larger, ~11% degradation, better-supported claim. Graph-construction comparison table shows backbone+temporal fusion (11139) beats geography-only (12851, −13%), gravity-only (10944, note gravity-only is actually *better* than the full fused model by MAE though the paper frames the comparison as "−13% vs gravity" using RMSE or a different base — worth a careful re-check if this number is cited), PCC-based (12260), backbone-only (12236), temporal-only (12487). Also reports information-entropy and mutual-information analysis of the backbone vs. temporal graphs (backbone lower entropy 3.5-4.2 bits vs. temporal 5.8-6.4 bits) framed as interpretability evidence that the two graphs capture complementary signal.

**I. Writing.** Includes explicit complexity/efficiency section: parameter count ~50% of MPSTAN; training speed 0.495 s/epoch vs. MPSTAN 33.6 s/epoch (~68× faster) — a genuinely quantified and favorable efficiency claim, consistent with DLinear's non-recurrent design. Limitations acknowledged directly: "as the forecasting window increases, the model faces challenges in accurate long-term forecasting" (though claims it still outperforms alternatives at that regime).

**J. Red flags.** (1) Paper's own acronym (BDGSTN) differs from the acronym used to name the group/repo (BDSTGNN) — a naming trap for anyone citing from memory. (2) No Cola-GNN/EpiGNN baseline despite superficial "successor" framing — it competes against STAN/MPSTAN/CovidGNN instead, on different (COVID, not ILI) datasets. (3) Shipped code's actual split mechanism (fixed 50/50 windows, hard-coded seed 1234, no CLI horizon flag) diverges from the paper's stated 60/20/20 chronological description — a real reproducibility gap discoverable only by reading the code, not the paper. (4) The "gravity-based" graph baseline outperforming the paper's own proposed fused graph by MAE (10944 vs. 11139) in the ablation table, while the accompanying text frames the comparison as a "−13%" improvement relative to gravity — this specific number deserves a second look before citing, as read literally the fused model is *not* the best row in that table by MAE.

**K. Relevance.** Good illustration of the divergence between what a paper's prose claims about its protocol and what its actual reference implementation does — directly useful ammunition for a "corrected evaluation protocol" paper, and a concrete instance (not hypothetical) of a paper's own ablation table containing a baseline that beats the proposed method on the reported metric while the surrounding text doesn't flag it.

---

### 7. MSGNN

**A. Bibliographic.** Mingjie Qiu, Zhiyi Tan, Bing-kun Bao. arXiv:2308.15840 (submitted 30 Aug 2023); Data Mining and Knowledge Discovery (2024) per task brief and confirmed by the abstract-page venue field. No code link found in the fetched content.

**B. Claimed contributions (paraphrase; abstract is truncated in what was retrievable).** Targets two stated gaps: "(1) single-scale models fail to preserve long-range connectivity between distant epidemic-related areas, and (2) multi-scale epidemic patterns are ignored." Proposes a graph-structure-learning module plus multi-scale graph convolution that separates scale-shared from scale-specific patterns, across county and state administrative levels simultaneously.

**C. Method (5-8 lines).** A Graph Learning Module: N-Beats-style temporal convolution blocks with location embeddings; a long-range block models state-level connectivity via a learned similarity `A^{l,k}_s = f[θ_s^T · Concat(H_l, H_k)]`; a short-range block models county-level dependencies using geographic distance + administrative-boundary features. A Multi-scale Graph Convolution Module then applies scale-specific GCN message passing separately at micro (county) and macro (state) scales, a scale-shared pattern learner using temporal attention across scales, and a fusion block combining local/regional features via attention-weighted aggregation.

**D. Data & protocol (quoted).** **This paper does not use the Cola-GNN benchmark datasets at all.** It uses JHU CSSE COVID-19 case counts (Mar 1 2020–Jul 1 2021) at two US administrative levels simultaneously: 50 states (macro) and 3,142 counties (micro), evaluated specifically on the 500 and 100 most-populous counties, Feb-Jul 2021. Normalization by population (per-capita), not the mean/std normalization used by Cola-GNN/EpiGNN. Lookback window 14 days; horizons 1/2/3-week-ahead. Metrics: MAE, MAPE, RMSE on weekly confirmed cases. "Multiple random seeds with result averaging," batch size 4, lr 1e-3 — no exact seed count or std reported in what was fetched.

**E. Baselines.** Exclusively CDC Forecast Hub-style epidemiological ensembles/models: COVIDhub-ensemble, CU-nochange, Microsoft-DeepSTIA, USC-SI_kJalpha, UVA-Ensemble, Google_Harvard-CPF, CEID_Walk, IowaStateLW-STEM, JHUAPL-Bucky. **Neither Cola-GNN nor EpiGNN appears anywhere in the baseline list or results tables** — explicitly confirmed by the fetch ("Note: ColaGNN or EpiGNN [are] not include[d] in results tables").

**F. Headline numbers.** Not populated for the requested Japan-Prefectures/US-Regions/US-States/Australia-COVID table — **this dataset/benchmark simply does not appear in the paper.** For reference, MSGNN @500 counties: 1wk/2wk/3wk MAE = 121.3/502.2/959.6 vs. COVIDhub-ensemble 132.2/550.2/1040.5. @100 counties: 321.5/1360.4/2588.5 vs. COVIDhub-ensemble 343.8/1482.2/2808.6.

**G. Improvement magnitudes.** ~8-11% MAE reduction vs. COVIDhub-ensemble (a real operational forecasting-hub baseline, not a re-implemented deep model) across horizons — a comparison against a strong, independently-operated ensemble is methodologically more honest than beating a hand-tuned re-implementation of a prior paper's model, though no significance test is reported.

**H. Ablations (MAPE, @100 counties).** Full model 0.428; "w/o multi-scale" 0.509 (+8.1%); "w-GCN" (presumably standard GCN instead of the paper's learned fusion) 0.470 (+4.2%); "w-GAT" 0.479 (+5.1%); "w/o fusion" 0.467 (+3.9%). Consistent, monotonic degradations support the architectural claims reasonably well.

**I. Writing.** Weekly-resolution robustness analysis (Feb-Jul 2021) reports error distributions (medians, variation, outliers) across time rather than just a single aggregate table — a better practice than most papers in this batch for showing temporal robustness, though still not a formal significance test. Limitations: not explicitly discussed in the retrievable content.

**J. Red flags.** The central one: **MSGNN is not actually part of the "Cola-GNN benchmark family" at all** — it is a COVID-19 US county/state hotspot-forecasting paper benchmarked against CDC Forecast Hub models, sharing only the general "epidemic + GNN + multi-scale" theme with the rest of this group. Grouping it with HeatGNN/EARTH/EpiHybridGNN/STTGNN risks implying a false apples-to-apples comparability that does not exist in the source literature.

**K. Relevance.** Low direct relevance for numeric benchmark comparison (wrong datasets, wrong baselines), but useful as a cautionary data point for your related-work section: not every "successor" architecture actually evaluates on the Cola-GNN/EpiGNN four-dataset benchmark, and citation chains that imply otherwise should be checked paper-by-paper exactly as done here. Its multi-scale-graph-vs-single-scale ablation (+8.1% from removing multi-scale) is a legitimate independent data point that some cross-scale structure helps, if you want a citation for that general claim outside the core four-dataset lineage.

---

### 8. EASTG and CSTGNN — not independently readable

**EASTG (Xu et al., IEEE BIBM 2025).** Could not locate a paper matching this exact acronym. The closest matching real paper found is Xu, M., Liu, Y., Guo, S., Liu, Y., Gao, C., "Enhancing Infectious Disease Forecasting via Epidemiology-Informed Adaptive Spatio-Temporal Graph Neural Networks," IEEE BIBM 2025, DOI 10.1109/BIBM66473.2025.11356232, pp. 1383-1390, online 15 Dec 2025 — but this paper's own acronym (per a search-engine-indexed description of its abstract) is **EISTGNN**, not EASTG, and it is validated on **China provincial-level and Germany state-level** data, not the Cola-GNN Japan/US/Australia benchmark. Full text sits behind IEEE Xplore; I could not fetch it (no open-access copy, no arXiv preprint found via search). Given the acronym mismatch and the different dataset, I cannot confirm this is the paper the task intends, and I am not fabricating numbers for it. If this is indeed a different, not-yet-indexed BIBM 2025 paper literally named "EASTG," I was unable to find it under that name via WebSearch.

**CSTGNN (Han et al., 2025).** Found only via secondary description: a review paper ("From graph models to intelligent decision-making: a review of spatio-temporal graph neural networks for regional disease risk prediction and etiology mining," PMC13438451) mentions "CSTGNN, proposed by Han et al., combined physics-based SIR modeling with learnable spatio-temporal GCN embeddings, delivering both accurate predictions and interpretable epidemiological parameters (e.g., reproduction number R0 and adaptive contact matrices)," attributed to a 2025 ResearchSquare preprint by Han X, Wang Y, Zhang L, Li M, Chen J, Liu S, titled "CSTGNN: a hybrid approach for epidemic forecasting combining physics-based modeling and graph neural networks." I could not locate or fetch the primary ResearchSquare page itself (no direct URL surfaced by search), so everything reported about CSTGNN here is second-hand, via a third party's review summary, not a direct read of the paper. No datasets, protocol, baselines, or numeric results can be responsibly reported for CSTGNN — I am explicitly not filling this in.

**K. Relevance (both).** Neither can be assessed for methodology rigor or benchmark comparability from what is accessible. Both nonetheless reinforce a broader observation across this whole group: physics-informed SIR-hybrid GNNs are now the dominant new sub-genre (HeatGNN, EARTH, PISID, BDGSTN, and apparently both EASTG/EISTGNN and CSTGNN all bind an SIR-style mechanistic core to a GNN or graph-free encoder) — worth naming as a trend in the related-work section even where individual papers can't be cited with full numeric confidence.

---

### Cross-paper synthesis (~400 words)

Claimed gains in this group are large on paper — routinely 10-50% RMSE/MAE reduction over "the best baseline" — but almost none of that margin survives contact with the actual evaluation protocols once compared line by line. Every paper that reports Cola-GNN or EpiGNN numbers does so under a *different* window, split fraction, horizon grid, and normalization than either source paper used, and in the two cases checked in detail (HeatGNN, EARTH) the re-run Cola-GNN/EpiGNN numbers are substantially *worse* than those models' own self-reported numbers — EARTH's re-run puts Cola-GNN at 299.1 RMSE on US-States h=5 versus Cola-GNN's own 202, and EpiGNN at 288.5 versus EpiGNN's own 160. That is not evidence of a stronger new architecture; it is evidence that "beats Cola-GNN" claims in this literature are frequently measuring a re-implementation gap, not an architecture gap. Split fractions alone vary from 60/20/20 (HeatGNN, BDGSTN's prose) to 50/20/30 (EpiHybridGNN) to 6:1:3 (PISID) to fixed-window, non-fractional splits that diverge from a paper's own prose description (BDGSTN's shipped code). Horizon grids vary from {2,5,7,12} to {2,3,5,10,15} to {5,10,15} to {2,5,8,...,32} to {14,28} days — nominally "the same h=5" is not the same forecast task across papers. Worse, two of the eight papers in this "successor" group (MSGNN, and effectively BDGSTN/PISID by dataset choice) do not evaluate on the Cola-GNN four-dataset benchmark at all, despite superficially belonging to the same lineage by theme and citation graph — MSGNN benchmarks against CDC Forecast Hub ensembles on county-level COVID data, and BDGSTN/PISID use COVID daily series rather than the ILI-era weekly series, with neither including EpiGNN as a baseline. Graph-free models do hold their own: PISID's ablation directly shows a learned spatial-identity embedding (no adjacency) beating GWNet, ColaGNN, and several other graph/sequence encoders, and its explicit motivation — that graph structure often adds complexity without proportionate benefit — is the strongest existing statement in this literature of the argument your new paper wants to make. Recurring writing patterns: SIR-hybrid "physics-informed" framing is now the dominant novelty lever (5 of 8 papers); "Limitations" sections are rare (only PISID and, briefly, BDGSTN have one); variance/seed reporting is inconsistent (PISID is the only one with std in its main table; HeatGNN and EARTH both average 5 seeds but report no spread); and no paper in this group runs a formal significance test on its headline comparison, leaving every "outperforms" claim numerically unfalsifiable as stated.

---

## G3 — Benchmarks, evaluation critiques, operational evidence

Close reading for MSAGAT-Net Paper B (evaluation-critique companion to the renewal-decoder
paper). Read via arXiv HTML / PDF and web search where HTML unavailable. Access limitations
are flagged explicitly per paper (see §8, Cramer et al.).

---

### 1. SpatialEpiBench (Lyu, Turcan, Wilder, arXiv 2605.06530, May 2026)

#### A. Bibliographic + links
Ruiqi Lyu, Alistair Turcan, Bryan Wilder. "SpatialEpiBench: Benchmarking Spatial Information
and Epidemic Priors in Forecasting." arXiv:2605.06530v1, submitted 7 May 2026, CC BY 4.0.
HTML: https://arxiv.org/html/2605.06530v1. Code+data: github.com/Rachel-Lyu/SpatialEpiBench.
Read: full HTML (introduction, methods, results, discussion, appendices).

#### B. Main claims (quoted)
- "Geographic adjacency may be a poor proxy for cross-region dependencies and can hinder
  performance."
- Three failure modes: "(1) poor outbreak anticipation, (2) difficulty handling sparsity and
  noise, and (3) limited utility of common geographic adjacency."
- "Every method beats the naive baseline less than 50% of the time, often far less."
- "Almost no method outperforms naive. All but two methods perform significantly worse than
  naive during outbreaks."

#### C. Evaluation design
- **11 datasets**, all US/Canada/Australia (ILINet US-Regions flu; NCHSdeaths; CANpositivity;
  CHNGinpatient/outpatient; CPRadmissions; DVcli; HHShosp; JHUCase [US COVID]; CAcase [Canada];
  AUcase [Australia COVID]). **No overlap with Japan-Prefectures or UK NHS/LTLA.** AUcase
  overlaps in spirit with our Australia-COVID dataset (different source/geography level, not
  verified identical).
- **Rolling-origin**, retrained every 8 steps, training window = most recent 100 obs, 80/20
  chronological train/val, evaluated on the next 8 origins after each retrain. Horizons h∈{1..4}
  weekly, h∈{1..28} daily.
- **Metrics**: point only — MSE/MAE/RMSE/medAE/medSE, filtered variants (excl. zeros/outliers),
  and a "win rate" (fraction of origins beating the naive baseline). Explicitly **no
  probabilistic metrics** ("most benchmarked methods produce only point estimates").
- **Naive baseline**: last-observation persistence, per region, all horizons. Also DLinear and
  ARIMA(1,0,0) per node as univariate baselines.
- **Significance testing**: bootstrap 95% CIs, "bootstrapping across months and meta-analyzing
  across horizons." No seeds/multiple-run variance reported; single rolling protocol per model.

#### D. Models compared / headline results
Compared: DCRNN, AGCRN, STGCN, GraphWaveNet, MTGNN, GTS, StemGNN, STNorm (general STGNNs);
**EpiGNN, Cola-GNN, EARTH** (epidemic-specific, EARTH described as "EpiColaGNN" = Cola-GNN +
NGM patch); DLinear, ARIMA, naive persistence; and their own **TusoAI** (agentic architecture-
search method). **No foundation models tested** (no Chronos/TimesFM/Moirai/TabPFN-TS).
- Best epidemic-prior patches: "lower RMSE by an average of 9% and increase win rates by 15%"
  relative to un-patched models — still mostly losing to naive on win-rate.
- TusoAI "outperforms naive on 6 datasets, including 4 where the best adjacency-informed method
  did not."
- Filtered metrics (dropping zeros/outliers) change results 10–25% vs raw RMSE — a methodological
  warning about metric fragility.

#### E. Recommendations (quoted)
"[We] omit probabilistic metrics such as CRPS and WIS because most benchmarked models produce
point forecasts... we do not study auxiliary features or richer spatial networks, since such
data may be unavailable... we focus mainly on ILI and COVID-19." Frames the gap: "spatiotemporal
epidemic forecasting still lacks an independent standardized benchmark... it remains difficult
to determine whether spatial information and epidemic priors genuinely improve forecasting
utility."

#### F. Writing/framing
Standard ML-venue structure: Intro, Related Work, Methods (benchmark design + 4 "epidemic
priors" patches), Experimental Results, Discussion, References, heavy appendices. Frames itself
explicitly as an independent, standardized benchmark filling a gap (cf. missing "ImageNet" for
epidemic forecasting), with public code/data as the deliverable, not a new SOTA model. TusoAI is
included almost as a proof-of-concept that automated search beats hand-designed GNN adjacency
priors, not the centerpiece.

#### G. Relation to our paper
Directly supportive of our floor claim: independent benchmark across 11 US/CA/AU epidemic
datasets finds GNN spatial priors net-negative to neutral against naive persistence, aligning
with our Japan finding that seasonal/naive floors beat trained GNNs. It goes further than us in
breadth (11 datasets, rolling-origin) but is *shallower* on inferential rigor — no DM test, no
power/MDE analysis, no seeds. Our paper's contribution of an explicit power/MDE analysis showing
test sets can't resolve 4-6% gains is not present here at all; this paper's bootstrap CIs are the
nearest analog but not framed as a power argument. It doesn't touch protocol-correctness bugs
like lead-h scoring, and doesn't propose graph-free scale-equivariant models — they lean toward
"drop the graph prior, use priors/AutoML" rather than a principled architecture change. Largely
complementary, not redundant: cite as corroborating evidence at larger scale; note we add rigor
(significance/power) SpatialEpiBench lacks.

---

### 2. EpiCastBench (Panja, D'Agostino, Li, Chakraborty, Liu, arXiv 2605.11598, May 2026)

#### A. Bibliographic + links
Madhurima Panja, Danny D'Agostino, Huitao Li, Tanujit Chakraborty, Nan Liu. "EpiCastBench:
Datasets and Benchmarks for Multivariate Epidemic Forecasting." arXiv:2605.11598v1, 12 May 2026.
HTML: https://arxiv.org/html/2605.11598v1. Data: kaggle.com/datasets/aimltsf/epicastbench.
Code: github.com/aimltsf/EpiCastBench. Read: full HTML.

#### B. Main claims (quoted)
"To ensure reproducibility and fair comparison, we establish standardized evaluation settings,
including a unified forecasting horizon, consistent preprocessing pipelines, diverse performance
metrics, and statistical significance testing." Positions the gap as: "epidemic forecasting
domain lacks comparable benchmark datasets" (analogy to ImageNet/GLUE).

#### C. Evaluation design
- **40 datasets**, 8 diseases, 27 regions: COVID-19 (21, incl. **Australia [NSW], US [NYT], UK,
  Japan** — national, not prefecture-level), Dengue (9: Argentina, Brazil, Colombia, Malaysia×2,
  Panama, Peru, Philippines, Taiwan), Zika (2), Influenza (2, US), Chikungunya (2), Tuberculosis
  (2, incl. Japan), Measles (1, US), Chickenpox (1, Hungary). **No Japan-prefecture granularity,
  no ILINet regional flu, no NHS/LTLA.**
- Rolling-window, fixed random state 42, no task-specific tuning. Three horizon settings: long
  (30d/12wk/24mo), medium (14d/8wk/12mo), short (7d/4wk/6mo). Spain COVID excluded from
  long-term due to short series.
- **Metrics**: point only (MAE, RMSE, MASE, sMAPE), per-series then averaged.
- **Naive baseline**: random-walk/persistence, Y_t = Y_{t-1}.
- **Significance testing**: Friedman rank test per horizon/metric (all p<0.01) + post-hoc MCB
  (Multiple Comparisons with the Best) via critical distance. **No seed sensitivity reported**
  beyond the fixed seed 42.

#### D. Models / headline results
15 models: Naive, DLinear, Random Forest, XGBoost, TSMixer, KAN, LSTM, DeepAR, TCN, NBeats,
NHiTS, Transformer, TiDE, **Chronos-2**, **TimesFM**. **No Cola-GNN/EpiGNN/EARTH, no
spatiotemporal GNNs, no mechanistic models.** Foundation models dominate: "foundation models,
particularly TimesFM and Chronos-2, demonstrate superior performance, consistently achieving the
lowest median errors." Example short-term MASE: India COVID TimesFM 0.240 (best) vs Chronos-2
0.361; Taiwan Dengue TimesFM 0.151 (best); naive avg rank 10.29 vs TimesFM 2.43 / Chronos-2 2.04.

#### E. Recommendations (quoted)
"Although EpiCastBench is designed for multivariate forecasting, it does not explicitly capture
spatial dependencies and can be extended to spatiotemporal settings to model interactions in
disease transmission across regions." "The current setup focuses on deterministic forecasting
methods and evaluation metrics, without incorporating probabilistic forecasting... incorporating
probabilistic forecasting is an important direction... under uncertainty in public health
settings."

#### F. Writing/framing
~15-page benchmark paper: Abstract, Intro, Related Work, §3 Epidemic Dataset (radar-plot feature
analysis), §4 Baseline Evaluation (setup/performance/significance), §5 Conclusion+Limitations,
Appendix (baseline descriptions, metrics, full result tables). Explicitly positions itself via
the ImageNet/GLUE analogy — "benchmark as infrastructure" framing, heavy on breadth (40
datasets/8 diseases) over depth of any one evaluation axis.

#### G. Relation to our paper
Useful counterpoint: shows foundation models (Chronos-2, TimesFM) reliably beat naive across 40
datasets/8 diseases when GNNs are absent from the pool — complements SpatialEpiBench's finding
that GNNs *don't* beat naive. Neither paper puts foundation models and GNNs in the *same*
head-to-head, which is a gap our paper doesn't fill either but should flag. Their Friedman+MCB
significance testing is a genuine, useful method absent from most of this literature — closer to
our DM-test approach in spirit but coarser (rank-based across all datasets, not a per-comparison
power/MDE argument). No power/MDE analysis. No lead-h protocol discussion. No graph-free
scale-equivariant proposal. Complementary evidence, not redundant.

---

### 3. "A Critical Audit of Spatiotemporal Forecasting Benchmark Datasets and Baselines"
(Martin, Heilig, Fischer, Haddad, Sykulski, Eliasof, arXiv 2608.20980, Aug 2026)

#### A. Bibliographic + links
Kenneth Martin, Simon Heilig, Asja Fischer, Michel F. C. Haddad, Adam M. Sykulski, Moshe
Eliasof. arXiv:2608.20980v1. HTML: https://arxiv.org/html/2608.20980v1. No code repo found in
the page content. Read: full HTML.

#### B. Main claims (quoted)
"Spatially-unaware linear models pose a stronger competitor than previously reported." "The
evaluation protocols contain baselines spanning from historical averages to classical machine
learning approaches. These baselines often show competitive performance compared to GNNs." On
data pipeline bugs: "the official data-loader from the PyTorch Geometric Temporal library
releases first-order differenced data for Chickenpox and PedalMe, which obscures the low
frequency signal"; "models trained on overdifferenced datasets will in general tend to overfit
or misfit the data."

#### C. Evaluation design
- **Datasets**: Chickenpox (Hungary, 20 nodes, weekly — the only epidemic dataset), PedalMe,
  WikiMaths, METR-LA, PEMS-BAY (non-epidemic traffic/other). **No overlap** with
  Japan-Prefectures, US-Regions, Australia-COVID, or UK NHS/LTLA.
- Methodology is primarily a **statistical audit tool**, not a forecasting protocol per se:
  Pearson temporal autocorrelation, adjacency-weighted spatial correlation, *partial* spatial
  correlation (removes temporal component via least-squares projection) to isolate genuine
  spatial signal from confounded temporal signal, plus spectral-analysis proof that first-order
  differencing acts as a high-pass filter (amplifies noise, damps low-frequency signal), and a
  Ljung-Box white-noise test.
- Splits: 90/10 temporal (Chickenpox/PedalMe/WikiMaths), 7:1:2 (METR-LA/PEMS-BAY). Horizons:
  one-step for small datasets; 12-step (60 min) for traffic. Not full rolling-origin — single
  temporal split with lag windows = 3× input window for the correlation analysis.
- **Metrics**: MSE (small datasets), MAE/RMSE/MAPE% (traffic).
- **Baselines defined precisely**: persistence (ŷ=y_t); seasonal historical average (P=7 or 52);
  ridge-regularized **per-node AR(H)** (spatially unaware); **RidgeVAR** (single ridge over
  flattened lag matrix, fully connected linear — spatially *aware* but linear); DLinear; SARIMA
  with AutoARIMA order selection.
- **Significance testing**: **no formal hypothesis test on model comparisons** — reports ± SD
  over **10 random seeds**, and a separate linear-regression/R² analysis (R²=0.591) for a
  synthetic heterophily experiment.

#### D. Models / headline results
GNN zoo compared: A3T-GCN, AGCRN, DCRNN, DyGrAE, EvolveGCN-H/O, GC-LSTM, GConvGRU, GConvLSTM,
MPNN-LSTM, TDE-GNN, T-GCN, GMAN, MTGNN, GTS, STNorm, D2STGNN, STID, STEP, STAEformer, vs.
SARIMA/AutoARIMA/DLinear/LSTM/GRU-GCN. **No Cola-GNN, EpiGNN, or foundation models.**
- Chickenpox MSE: best reported prior SOTA TDE-GNN 0.533±0.011, but **SARIMA+LSTM-time
  0.713±0.009** and SARIMA+GCRN-GRU-time 0.722±0.010 beat plain ARIMA-time (0.728) only
  marginally, and none of the GNNs shown beat SARIMA hybrids cleanly once undifferenced.
- WikiMaths: SARIMA+LSTM-time 0.443±0.002 beats TDE-GNN 0.565±0.017.
- METR-LA/PEMS-BAY: SARIMA+GCRN-GRU hybrid is competitive with or beats STAEformer.

#### E. Recommendations (quoted)
"we recommend reducing the over-reliance on such datasets for method comparison, and instead
advocate for more rigorous statistical evaluation." "the PedalMe and Chickenpox datasets should
remain undifferenced when used for benchmarking purposes." "GNNs might perhaps... implicitly
assume... homogenous spatiotemporal response to past signal, unlike... linear models." Caveat on
their own method: "they [correlations] are limited to capturing linear dependency structures."

#### F. Writing/framing
~8,000-word main text + 4 appendices (SARIMA details, re-evaluations, synthetic experiment,
settings). Explicitly self-styled as **"A Critical Audit"** — a statistical/methodological
deconstruction paper, not a new-model paper: §1 Intro (motivates critique of 5 widely-used
benchmarks), §2 correlation analysis + protocol critique, §3 baseline re-evaluation, §4 outlook +
synthetic heterophily experiment, §5 conclusion. This is the closest structural analog in the
group to "a paper whose contribution is a critique + re-evaluation," useful as a structural
template for our own evaluation-critique framing.

#### G. Relation to our paper
Strongly analogous in spirit to our Paper B: both argue benchmark GNN gains are partly artefacts
of weak baselines/broken pipelines rather than genuine spatial modelling. Their focus (traffic +
one epidemic dataset, data-loader differencing bug, correlation-based diagnostics) is
*methodologically adjacent but empirically disjoint* from ours (Japan/NHS/LTLA/Australia epidemic
panel, lead-h scoring bug, DM test + power analysis). No power/MDE analysis; no formal
significance test on their own head-to-head numbers (surprising given the critique framing — a
gap they leave for us). No naive-vs-GNN framing on epidemic data specifically (Chickenpox is
their only epidemic series, and it is UK-adjacent-genre, i.e. incidence data, but not our
geographies). Does not propose graph-free scale-equivariant architectures — recommends better
statistical diagnostics and undifferenced data, not a new model family. We can cite this as the
strongest published precedent for "checking whether a benchmark's own data pipeline biases
against baselines," and as independent corroboration that GNN architectural complexity is
frequently confounded with weak/broken baseline implementations — but we add rigor (formal DM
test + power) that even this self-styled "critical audit" lacks.

---

### 4. M-SPICE (Gomez, Wu, Wang, Shen, Rodríguez, KDD 2026, arXiv 2606.22171)

#### A. Bibliographic + links
Diana Guadalupe Gomez, Chenwei Wu, Zhiyi Wang, Liyue Shen, Alexander Rodríguez. Actual title per
arXiv metadata: **"Beyond Time Series: Spatial Reasoning for Epidemic Forecasting via Multimodal
Learning"** (the task brief's short name "M-SPICE" is the model name inside the paper, not the
title). arXiv:2606.22171. PDF: arxiv.org/pdf/2606.22171; HTML (experimental):
arxiv.org/html/2606.22171v1. No code link found in fetched content. Read: full HTML.

#### B. Main claims (quoted)
Core contribution: moving "beyond atomic regional representations" via "structure-aware
multimodal epidemic forecasting," with M-SPICE performing "joint reasoning over temporal disease
dynamics and spatial context via attention-based multimodal fusion."

#### C. Evaluation design
- **This is primarily a new-model paper, not a benchmark paper** — included in our group because
  it does rolling-origin, multi-disease evaluation with real operational framing.
- Datasets: US-state-level COVID-19/influenza/ILI (ILINet, RESPNet/CDC hospitalizations), plus
  auxiliary non-target signals (Google symptom search, ERA5 temperature, CDC/Michigan MDHHS
  county-level hospitalization maps as spatial context, not targets). **No Japan, Australia, or
  UK NHS/LTLA data.**
- **Rolling-origin**: train 2020W43–2023W39, evaluate over forecast origins 2023W40–2024W12,
  H=4 weeks. Explicitly *not* using vintage/real-time-available data (flagged as a limitation).
- **Metrics**: point (NRMSE, min-max normalized per region) and probabilistic (empirical 90%
  coverage, **relative WIS vs. persistence**, via post-hoc Adaptive Conformal Inference
  calibration).
- **Naive baseline**: persistence ("predicts future values using the most recently observed
  target"), explicitly named as the CDC-style reference for rWIS.
- **Significance**: 3 seeds [5, 17, 33], mean NRMSE reported; a few Pearson correlation
  significance tests on *auxiliary regional analysis* (e.g., r=0.551, p<0.001, for neighbor-count
  correlation) but **no formal significance test on the headline model-vs-baseline comparison**.

#### D. Models / headline results
Baselines: ARIMA, GRU, Autoformer, Crossformer, TimeMixer (temporal-only); Maestro, DiPro
(multimodal); **Persistence, Cola-GNN, CNNRNN-Res** (epidemic-specific — note CNNRNN-Res appears
here too, same baseline family as our repo uses); **EARTH attempted but excluded**: "unable to
[run] due to unresolved implementation issues" — a notable admission of baseline
reproducibility failure; **EpiGNN not included**.
- COVID NRMSE avg W1–4: M-SPICE (TS+Spatial) 0.194 vs TS-only 0.218 vs TimeMixer 0.228 vs
  Persistence 0.213 — i.e., **persistence beats the best temporal-only deep baseline (TimeMixer)
  on COVID**, and M-SPICE's full model only edges out persistence by ~9%.
- Flu NRMSE avg: M-SPICE 0.220 vs Maestro 0.230 vs Persistence 0.240.
- ILI NRMSE avg: M-SPICE 0.210 vs Maestro 0.217 vs Persistence 0.242.
- Gains concentrated at longest horizon (W4): COVID 9%, Flu 7%, improvement of spatial module
  over TS-only.
- rWIS, COVID: M-SPICE (TS+Spatial) 0.856 vs Persistence 1.000 (i.e., only ~14% probabilistic
  improvement over naive).

#### E. Recommendations (quoted)
"Although we simulated a real-time forecasting environment in our experimental setup, we did not
rely on vintage data." "performance may vary across regions due to differences in surveillance
quality... We do not explicitly model or correct for such disparities." "model outputs should be
interpreted as decision-support tools rather than definitive predictions in high-stakes public
health settings." Notes MetroCast Hub sub-state HSA data as a future evaluation resource.

#### F. Writing/framing
KDD-style applied-ML paper: Intro, Related Work, Problem Formulation, Methodology (spatial
encoder, temporal encoder, joint attention, horizon-dependent gating), Experiments (RQ1 does
spatial help? RQ2 robustness, RQ3 interpretability), Limitations & Ethics. Framed as a new model
+ attention-interpretability analysis, with benchmark rigor (rolling-origin, multi-disease,
multi-seed) as secondary/supporting evidence, not headline contribution.

#### G. Relation to our paper
Directly relevant precedent for our attention-inertness finding: even a paper *advocating* for
spatial attention only demonstrates 7–9% gains over persistence, concentrated at the longest
horizon, with modest rWIS improvement (14%) — margins in the same range our paper argues test
sets can't resolve statistically (their 4-6% MDE claim would likely flag M-SPICE's headline
numbers as statistically fragile too, though M-SPICE reports no significance test to check this).
EARTH failing to run is independent evidence that this baseline family is fragile/hard to
reproduce, consistent with anything we've found about baseline reproducibility. No power/MDE
analysis, no lead-h protocol critique, no graph-free proposal (M-SPICE moves toward *more*
spatial complexity, i.e. the opposite direction from our graph-free proposal). Good citation as
"even proponents of spatial modules report gains at the edge of statistical detectability,
unverified for significance" — this is an opening for our MDE argument.

---

### 5. "From naive to foundation" (Wang, Li, Perra, QMUL, medRxiv 2026)

#### A. Bibliographic + links
Wang, Li, Perra (Queen Mary University of London). medRxiv 10.64898/2026.05.11.26352889v1.
URL: https://www.medrxiv.org/content/10.64898/2026.05.11.26352889v1.full. Read: full text via
fetch (methods/results/discussion sections retrieved).

#### B. Main claims (quoted)
"The naive model acts as a critical diagnostic to ensure that complex models are genuinely
anticipating transmission dynamics rather than merely echoing recent observations." "Models
trained exclusively on real data exhibit high variance and frequent underperformance, often
failing to beat the naive baseline." "Without any task-specific retraining, it [TabPFN-TS]
successfully overcomes extreme data scarcity to consistently outperform all other individual
architectures." "Future research should focus on designing foundation models pre-trained
exclusively on massive epidemiological datasets."

#### C. Evaluation design
- **9 European countries** (Belgium, Czechia, Denmark, France, Ireland, Italy, Netherlands,
  Poland, Romania), ERVISS (ECDC/WHO-Europe) weekly ILI per 100k. Seasons 2017–18, 2018–19,
  2023–24 for training/val; **2024–25 held out as unseen test**. COVID seasons excluded.
  **No overlap** with Japan-Prefectures/US-Regions/Australia-COVID/UK NHS-LTLA (this is
  continental-Europe ILI, not UK).
- **Rolling-window**, weekly advance from ISO week 45 through week 14, H∈{1,2,3,4}. Cumulative
  window for ARIMA/SEIR/TabPFN-TS; fixed 4-week sliding window for LSTM/DLinear/Autoformer.
- **Metrics**: point (MAE, WMAPE) and probabilistic (80% interval score, IS₈₀). No CRPS/WIS
  terminology used (IS₈₀ is the *unweighted* interval score at one nominal level, narrower than
  full WIS).
- **Naive baseline**: exact persistence, ŷ_{t+h}=y_t for all h. "Used as reference denominator for
  relative performance metrics."
- **Significance testing: none reported** — no p-values, CIs, or formal tests. **No seeds
  mentioned.** TabPFN-TS run zero-shot with "internal parameters... strictly frozen" during
  inference — this is stated as a way to *avoid* leakage via retraining, but **no explicit
  temporal-contamination/pretraining-corpus audit** is described (the paper does not check
  whether the ERVISS/ECDC test data appeared in TabPFN-TS's own pretraining corpus).

#### D. Models / headline results
Naive; RespiCast (ECDC operational multi-model ensemble hub — the "field" baseline, analogous to
FluSight); ARIMA (auto-ARIMA); age-stratified SEIR (ABC-SMC calibrated); LSTM, DLinear,
Autoformer (deep learning, tested with and without synthetic data augmentation); **TabPFN-TS**
(zero-shot foundation model); a performance-weighted ensemble. **No Cola-GNN/EpiGNN/EARTH, no
Chronos/TimesFM/Moirai.**
- Horizon-4 IS₈₀: TabPFN-TS beats RespiCast in Ireland (41.2 vs 55.5) and Italy (869.6 vs 1405);
  RespiCast still wins 6/9 countries at h=4 on both point and probabilistic metrics.
- TabPFN-TS wins 4/9 countries at h=1 (probabilistic); "competitive" at h=1–2 point forecasts.
- Data augmentation materially helps deep learning: Ireland Autoformer IS₈₀ drops from 119.7
  (real data only) to 73.3 (augmented).
- Individual (non-ensemble, non-augmented) deep learning models rarely beat naive/RespiCast.

#### E. Recommendations (quoted)
"Deep learning architectures are severely constrained by extreme data scarcity, typical in
epidemic forecasting." "In real-time public health scenarios, recent epidemiological
observations are frequently subject to reporting delays, right-truncation, and subsequent
backfill revisions." "Many forecasting models, especially those based on deep learning, are
optimized for long-horizon prediction accuracy on large-scale, generic datasets, whereas...
short-term forecasts (typically 1−4 weeks ahead) are the standard in epidemiology."

#### F. Writing/framing
Structured as Abstract → Intro → Results (model taxonomy, data, training, forecasting setup,
point/probabilistic accuracy, epidemic-trajectory case studies) → Discussion → Materials and
Methods (methods-after-results is a medRxiv/biomedical convention, not ML-venue convention).
Framed as a progression narrative ("naive → classical → mechanistic → deep learning →
foundation"), explicitly testing whether sophistication beats simplicity under real data
scarcity, closing with an ensemble as the pragmatic answer — an implicit rebuttal to "just use a
foundation model" as well as to "just use a GNN."

#### G. Relation to our paper
Very close in spirit to our "naive/seasonal floors beat trained models" finding, but on
different data (European ILI, not Japan) and with deep learning (LSTM/DLinear/Autoformer)
rather than GNNs as the underperforming class — still a temporal, non-graph deep learning class,
so it's one more independent confirmation that architectural complexity underperforms simple
baselines under epidemic data scarcity, this time also implicating a leading foundation model
(TabPFN-TS) as the one architecture that *does* consistently help, complicating a pure
"complexity never helps" narrative — worth noting since our paper's alternative is a graph-free
*equivariant* model rather than a foundation model; this paper would likely say that's one
plausible fix but that pretrained/zero-shot approaches are another, competing fix worth
discussing. No power/MDE analysis, no significance testing at all (weaker than us on rigor), no
lead-h protocol critique (not applicable — no spatial GNN in their comparison), no leakage audit
for TabPFN-TS despite explicitly claiming "frozen parameters" as sufficient (a gap we should flag
if we cite it: they assert no leakage but never test for corpus contamination, which our
literature-critique lens should call out explicitly rather than accept on faith).

---

### 6. Jafari, Fox, Fox, Marathe, Adiga (arXiv 2606.19560, June 2026)

#### A. Bibliographic + links
Alireza Jafari, Judy Fox, Geoffrey C. Fox, Madhav Marathe, Aniruddha Adiga. "Understanding Key
Features of Time Series Foundation Models from Epidemic Forecasting." arXiv:2606.19560. HTML:
https://arxiv.org/html/2606.19560v1. No code link found in fetched content (paper text mentions
releasing code/checkpoints, but URL not visible in the extracted content). Read: full HTML.

#### B. Main claims (quoted)
"A mixture-of-experts model that fuses multiple pretrained forecasters achieves the strongest
overall performance." "Many reported gains rely on a narrow and unrealistic experimental design:
a single short country-level ILI series, no spatial structure or revision effects, and very long
horizons of 24–60 weeks that diverge from CDC-style operational forecasting practice." "Language-
style time-series foundation models have important limitations for ILI forecasting... Quantization,
prompt reprogramming, or mapping numerical windows into language-like token spaces can... weaken
the inductive biases needed for short-horizon epidemic forecasting."

#### C. Evaluation design
- ILI: CDC ILINet, weekly, **10 US HHS regions**, ~20 years to early 2025. Influenza
  hospitalizations: weekly, region-level, lab-confirmed, ~3 seasons. Auxiliary pretraining
  corpora: COVID-19 weekly deaths by HHS region, TrafficL (Caltrans), M4 (generic ~100k series).
  **US HHS regions only — no Japan, Australia, or UK NHS/LTLA overlap.**
- **Two protocols**: (a) **temporal split** — chronological train/val/test within the same
  regions; (b) **spatial split** — train on a subset of regions, test on entirely unseen regions
  (explicit geographic-transfer test, notably rare in this literature group). **Strict
  revision-freeze policy**: "each series is frozen to the version available as of the forecast
  date, disallowing retrospective access to backfilled values" — real-time-faithful, a leakage
  safeguard most of this group's papers lack. Horizons H∈{1,2,3,4} weeks, matching CDC FluSight
  practice explicitly contrasted with "24-60 week" foundation-model literature norms.
  Retraining-frequency ablation: retrain_window ∈ {200,100,50,10} steps (Table V) — "more
  frequent retraining yields consistent, though gradually diminishing, gains."
- **Metrics**: MSE and Normalized Nash-Sutcliffe Efficiency (NNSE, bounded [0,1], 0.5 = predicting
  the historical mean). **No probabilistic metrics** — explicitly deferred as future work.
- **Naive baseline**: ARIMA(1,0,0)(1,0,0), refit per origin over a 104-week rolling history — note
  this is **not** a pure persistence/naive baseline (no simple last-value or seasonal-naive
  reported in tables); NNSE=0.5 is used as the "historical mean" reference point instead.
- **Significance/leakage**: "at least five independent runs" averaged (no seed values given, no
  variance/CI reported in extracted tables). **No formal significance testing** (no p-values). On
  contamination: the paper does **not** formally verify whether Chronos/Bolt pretraining corpora
  include ILI/flu-hospitalization data; it only notes generically that "different studies adopt
  heterogeneous data pipelines and report inconsistent rankings for key baselines" — awareness of
  the problem, no audit performed. This is the most direct hit on the task's contamination
  question, and the answer is: **no rigorous leakage/contamination check for foundation models,
  despite the paper's stated focus on foundation-model behavior.**

#### D. Models / headline results
ARIMA; LSTM (direct/iterative); TCN; VanillaTransformer; TFT; TimesNet; PatchTST; iTransformer;
TiDE; **Chronos-T5 (mini/small/base/large), Chronos-Bolt (mini/small/base)**; PatchTST/
iTransformer with domain-specific pretraining (M4/TrafficL/Epidemic/Hospitalization); **Time-LLM**
(GPT-2 backbone); their own **MultiFoundationCore (MFC)** mixture-of-experts (fuses PatchTST,
TSMixer, TFT, iTransformer, TimeLLM, VanillaTransformer via cross-attention); TinyLSTM (compact,
short-data regime). **No Cola-GNN/EpiGNN/EARTH** — explicitly named in related work but not
empirically compared, and **no Moirai/TimesFM/TabPFN-TS**.
- Multi-horizon ILI (Table I): MFC MSE 0.382/NNSE 0.864 (best); PatchTST-Hospitalization 0.412/
  0.857; PatchTST-M4 0.415/0.856; plain PatchTST 0.439/0.850; TimeLLM-GPT2 0.724/0.772 (much
  worse); Chronos-T5-base 0.683/0.791 (iterative, worse than direct-strategy transformers);
  ARIMA 0.962/0.739 (worst, iterative).
- 1-week (Table II): MFC 0.112/0.954 (best); LSTM-iterative 0.122/0.950 close second.
- Spatial (unseen-region) evaluation (Table III/IV): plain PatchTST best non-MFC at 1-week
  (0.166/0.935); MFC not listed as clearly dominant on spatial split in the extracted numbers —
  suggests generalization to unseen regions is harder even for the best temporal model,
  consistent with "the dataset contains only 10 regions, which limits diversity available for
  cross-region generalization."
- Hospitalization (Table VII): MFC+ILI 0.00224 (best); TinyLSTM+ILI 0.00237.

#### E. Recommendations (quoted)
"Improvements are largest at longer lead times (weeks 3–4) and are strongest when the pretraining
domain is mechanistically aligned with the target task." "A likely reason [models perform better
on temporal than spatial splits] is that the dataset contains only 10 regions, which limits the
diversity available for cross-region generalization." "Probabilistic forecasting, uncertainty
quantification, calibration analysis, and broader spatial generalization are important next
steps." "Integrating mechanistic or causal structure—such as SEIR-style components, mobility,
interventions, and demographic features—into foundation-style backbones may further improve
robustness under regime shifts and pandemics."

#### F. Writing/framing
Engineering/systems-style evaluation paper: Abstract, Intro, Related Work (mechanistic → deep
learning → TS foundation models), Data (temporal/spatial splits, revision-freeze preprocessing),
Model Evaluation (Temporal ILI → Spatial ILI → Retraining-frequency ablation → Hospitalization →
Scope/Limitations), Conclusion, Appendix (hyperparameters for 17 models). Explicitly positions
itself against "unrealistic" foundation-model benchmarks (single national series, 24-60wk
horizons) in favor of CDC-aligned, multi-region, 1-4wk practice — i.e., an operational-realism
critique of the FM literature, parallel in spirit to our protocol-correction argument but aimed
at a different community (FM papers vs. GNN papers).

#### G. Relation to our paper
The closest match in this group to our "operational realism vs. published FM claims" critique,
but for foundation models on flu rather than GNNs on Japan/UK data. Its temporal-vs-spatial split
distinction and revision-freeze policy are methodologically stronger than most of this list and
worth citing as best practice we can point to (and, if relevant, adopt language from) — our repo
already treats normalization-fit-on-train and chronological splits as non-negotiable (AGENTS.md
§4), so this paper's revision-freeze discipline is a natural complement to cite. It does **not**
run a formal significance/power test, and explicitly does **not** audit foundation-model
pretraining-corpus contamination despite naming that as a known confound — this is a genuine gap
our critique can note is still open even in the most careful FM-evaluation paper in the set. No
GNN baselines at all, so it cannot speak to the lead-h scoring bug. No graph-free
scale-equivariant proposal — but MFC (mixture-of-experts over non-graph temporal transformers) is
directionally aligned with "graph-free" architectures outperforming graph-based ones, useful
corroboration for a non-spatial model class beating the field.

---

### 7. Mathis et al., FluSight evaluation (Nature Communications 15:6289, 2024)

#### A. Bibliographic + links
Mathis et al. "Evaluation of FluSight influenza forecasting in the 2021–22 and 2022–23 seasons
with a new target laboratory-confirmed influenza hospitalizations." Nature Communications 15,
6289 (2024). DOI: 10.1038/s41467-024-50601-9. PMC: PMC11282251 (pmc.ncbi.nlm.nih.gov/articles/
PMC11282251/). Read: full text via PMC fetch (methods + results sections retrieved in detail).

#### B. Main claims (quoted)
"Forecast skill and 95% coverage for the FluSight ensemble and most component models degrade over
longer forecast horizons." "As the forecast horizon moved from 1 to 4-weeks, the FluSight
ensemble 95% prediction interval coverage declined from 89.61% to 83.74% in 2021–22 and from
85.69% to 77.85% in 2022–23. These results highlight room for improvement in model calibration,
as almost all models (with the exception of the UMass trends ensemble) were overconfident in
their predictions." "Forecasting remains difficult in periods of rapid change and epidemic
turning points."

#### C. Evaluation design
- **Baseline**: not simple persistence — a **"quantile baseline"**: "the median prediction of the
  baseline forecasts is the corresponding target value observed in the previous week, and noise
  around the median prediction is generated using positive and negative 1-week differences...
  for all prior observations, separately for each jurisdiction." I.e., persistence-in-median with
  an empirical-residual-based quantile spread — a genuinely probabilistic naive baseline, more
  sophisticated than plain last-value persistence used elsewhere in this group.
- **Metrics**: WIS ("a proper score that generates interval scores for probabilistic forecasts
  provided in the quantile format... to account for dispersion, underprediction, and
  overprediction"). **Relative WIS**: "computes the ratio of average WIS values for each pair of
  models on the subset of forecasts that both models provided, and then normalizes by the mean
  pairwise WIS ratio for the baseline model" — this is the **pairwise relative-skill method**
  from the Forecast Hub literature (same method as Cramer et al. §8 below), which handles
  unequal submission coverage across teams. Coverage = "percent of observed values that fall
  within the 50%/95% prediction intervals."
- Forecast origins: 18 weeks (Feb 21–Jun 20, 2022) and 30 weeks (Oct 17, 2022–May 15, 2023).
  Horizons 1–4 weeks. Locations: national + 50 states + DC + Puerto Rico (52 jurisdictions).
  Weekly submission cadence (no explicit "retraining" language — teams submit independently).
- **Models/ensemble**: 2021–22: 26 teams submitted, 23 eligible (median 20/week); 2022–23: 26
  submitted, 18 eligible (median 15/week). FluSight ensemble = **unweighted median of each
  quantile among eligible forecasts** (Forecast Hub convention).
- **Significance testing: none formal** — "no formal hypothesis testing, p-values, or bootstrap
  confidence intervals for comparing models. Results rely on rank-based comparisons and relative
  WIS ratios without inferential statistics." This is an explicit, notable absence in a
  high-profile Nature Communications operational-forecasting paper.

#### D. Headline numerical results
- Models beating baseline (relative WIS ≤1.0): 6/23 in 2021–22; 12/18 in 2022–23.
- FluSight ensemble relative WIS: 0.82 (2nd rank) in 2021–22; 0.77 (5th rank) in 2022–23 — **the
  ensemble is not always the top performer**, contra the common "ensemble always wins" framing
  found elsewhere (e.g., contrast with Cramer et al. §8, where the COVID ensemble was #1).
  Top individual models: CMU-TimeSeries (0.74 / 0.67), PSI-DICE (0.83 / 0.70), MOBS-GLEAM_FLUH
  (1.02 / 0.61 — went from worse-than-baseline to best across seasons).
- Coverage: ensemble 89.3%→83.3% (1wk→4wk) 2021–22; 85.8%→77.9% 2022–23 — systematic
  overconfidence, worsening with horizon.
- No Cola-GNN/EpiGNN/EARTH/foundation models — this is an operational human-team forecast hub
  evaluation, not a methods benchmark; none of our group's architectures appear here.

#### E. Recommendations (quoted)
"It may be possible that an ensemble of forecasts for categorical increases or decreases in
activity may have additional utility in terms of preserving valuable information while also
maintaining the benefits of the use of ensembles over individual models."

#### F. Writing/framing
Standard Nature Communications applied-epidemiology paper: methods rigorously describe the WIS/
relative-WIS/coverage machinery (these are now the field-standard metrics, inherited from the
COVID Forecast Hub practice established by Cramer et al.), then results by season/horizon/
jurisdiction, then discussion of calibration and operational implications. No "benchmark
contribution" framing at all — it's a retrospective operational evaluation of a live,
policy-relevant forecasting hub, written for a public-health/epidemiology audience rather than an
ML audience. Useful structural contrast: this is what "the real-world evaluation standard" looks
like when done by an operational consortium (CDC/Forecast Hub) rather than an ML lab — heavy on
calibration and coverage, essentially silent on significance testing.

#### G. Relation to our paper
Establishes the **field-standard WIS/relative-WIS/coverage methodology** that any epidemic
forecasting evaluation should be measured against — our paper (GNN point + calibration focus)
should be explicit about whether we use WIS/relative-WIS in the Forecast-Hub-standard sense or a
different probabilistic metric, and should cite this as the canonical definition source. The
explicit finding that **the ensemble is not always #1** (5th rank in 2022–23) is a useful,
citable caution against "bigger/combined model always wins" — relevant if our paper discusses
ensembling as a possible fix. The **total absence of significance testing** in a landmark,
policy-relevant Nature Communications paper is itself a strong argument for why our DM-test +
power/MDE contribution is novel and needed — this is exactly the gap our paper fills that even
the flagship operational forecast-hub evaluation doesn't. No relevance to lead-h scoring bugs or
graph-free architectures (no spatial models here at all).

---

### 8. Cramer et al., US COVID-19 Forecast Hub evaluation (PNAS 119(15) e2113561119, 2022)

#### A. Bibliographic + links
Cramer et al., "Evaluation of individual and ensemble probabilistic forecasts of COVID-19
mortality in the United States." PNAS 119(15):e2113561119, 2022. DOI: 10.1073/pnas.2113561119.
Correction: PNAS 120, pnas.2304076120 (2023).

**Access limitation — stated explicitly per instructions**: I could **not** obtain the full text.
Attempted: pnas.org/doi (403 Forbidden), pnas.org/doi/pdf and /epdf (403), medRxiv preprint
v2/v3 full/full.pdf (403 on all three URL forms), PMC (no PMC ID resolved via PubMed — PubMed
page itself returned only a cookie-consent stub, no article content), ResearchGate (403), OSTI
PDF (connection refused), Iowa State repository page (no full text, redirect only), Semantic
Scholar (empty page). **What follows is therefore built only from web-search result snippets and
the abstract, not the full paper text** — I did not read the Methods/Results/Discussion sections
directly. Any numbers below are as reported in search-engine summaries of the paper, not
independently verified against primary tables/figures.

#### B. Main claims (as reported in search snippets, not verified against full text)
"A multimodel ensemble forecast that combined predictions from dozens of groups every week
provided the most consistently accurate probabilistic forecasts of incident deaths due to
COVID-19 at the state and national level from April 2020 through October 2021 [some sources say
April 2021]." "An ensemble model provided a reliable and comparatively accurate means of
forecasting deaths... that exceeded the performance of all of the models that contributed to it."

#### C. Evaluation design (partial, from snippets only)
- COVID-19 Forecast Hub, US, collecting forecasts from "more than 80 different academic,
  industry, and independent research groups" starting April 2020. National + state-level. This is
  the direct precedent to the FluSight paper (§7) — same WIS/relative-WIS/pairwise methodology
  the Forecast Hub ecosystem standardized on.
- **I cannot state the exact number of forecast origins, horizons, retraining cadence, or the
  precise baseline definition from what I was able to access** — these require the full Methods
  section, which I did not obtain. (By close analogy to §7 and general Forecast Hub practice, the
  baseline is very likely the same "quantile persistence-with-empirical-residual-spread" design,
  but I am not asserting this as read; it is inference, not verified quotation.)
- **Robustness checks** (from snippet): "Values of relative WIS and rankings of models were
  robust to changing thresholds for submission inclusion criteria and to the inclusion or
  exclusion of individual outlying or revised observations" — this is a real methodological
  detail from the search summary, closer to a sensitivity analysis than a significance test, but
  I could not confirm exact method (bootstrap? leave-one-out?) from the snippet alone.

#### D. Headline numerical results (from snippets only)
"The COVIDhub-ensemble achieved a relative WIS of 0.61" (interpreted in the snippet as "39% less
probabilistic error than baseline on average, adjusting for difficulty of specific predictions").
"18 models had a relative WIS of less than 1... and 10 models (including the baseline) had a
relative WIS of 1 or greater" — implying **~28 models total** in the comparison set. No
per-horizon breakdown, no GNN/foundation-model comparison possible (this paper predates
Cola-GNN/EpiGNN's mainstream adoption in forecast hubs and long predates foundation-model time
series methods; the Forecast Hub pool is composed of independent human-team submissions, not a
controlled architecture bake-off).

#### E–F. Recommendations, writing/framing
**Not established from what I read** — I would be fabricating if I described discussion-section
recommendations or structural framing beyond what the abstract-level snippets state. What is
clear even from the abstract framing: like Mathis et al., this is an operational-hub retrospective
evaluation for a public-health audience (PNAS main track, not an ML venue), and the "ensemble beat
every individual model that composed it" claim is the headline result — a genuinely different
outcome from Mathis et al.'s finding that the FluSight ensemble was NOT always top-ranked (5th in
one season). This contrast (COVID ensemble = clear #1; flu ensemble = middling) is worth flagging
if citing both, but I have not verified the COVID claim beyond the search snippet.

#### G. Relation to our paper
Given the access limitation, I can only say with confidence: (1) this paper, together with
Mathis et al., established the WIS/relative-WIS field standard our own probabilistic evaluation
(if any) should be benchmarked against or explicitly justified as departing from; (2) it is widely
cited as the canonical demonstration that ensembling helps in operational COVID forecasting,
which is a useful contrast point to Mathis et al.'s more mixed flu finding, and to our own
skepticism about complexity (ensembling of independent human-team models is a different kind of
"more complexity" than a single trained GNN, so the two findings are not necessarily in tension
with our floor-beats-GNN result — worth being careful about this distinction if cited). I cannot
verify whether it performs any DM-style significance test, reports seeds, or discusses lead-h
protocol issues — **flagging this explicitly as unread rather than guessing**.

---

### Cross-cutting check: minimum detectable effect / statistical power

**None of the eight works computes a formal minimum-detectable-effect (MDE) or statistical-power
analysis for forecast comparisons.** Closest approaches found:
- SpatialEpiBench (§1): bootstrap 95% CIs on win-rate/RMSE, meta-analyzed across horizons — a
  variance estimate, not a power/MDE calculation.
- EpiCastBench (§2): Friedman test + MCB critical-distance post-hoc — controls for multiple
  comparisons and gives a formal "statistically indistinguishable from best" set, but does not
  report power or minimum detectable effect size.
- Critical Audit (§3): 10-seed ± SD only; explicitly *no* formal significance test on their
  head-to-head model comparisons, despite the "critical audit" framing — a genuine, notable gap.
- M-SPICE (§4), "From naive to foundation" (§5), Jafari et al. (§6): no significance testing at
  all on headline model comparisons.
- Mathis et al. (§7): explicitly states no formal hypothesis testing was used.
- Cramer et al. (§8): unverified (access blocked), but the snippet-level "robustness to inclusion
  criteria" is a sensitivity check, not a power analysis, as best I can tell.

**None of the eight runs Cola-GNN or EpiGNN under a corrected/rolling protocol with
per-horizon reporting** in the way our paper does. Cola-GNN appears in SpatialEpiBench (§1, one
of many GNN baselines, rolling-origin, but headline results are pooled win-rate/RMSE, not broken
out per-horizon in what I extracted) and in M-SPICE (§4, one baseline among several, results
reported as NRMSE averaged W1–4, not per-horizon in the extracted tables). EpiGNN appears only in
SpatialEpiBench. EARTH appears in both SpatialEpiBench and M-SPICE, and M-SPICE explicitly could
not get it running ("unresolved implementation issues") — independent evidence of baseline
fragility in this literature. **No paper in this group reports Cola-GNN/EpiGNN broken out by
individual horizon (h=1,2,3,4,...) with per-horizon significance testing**, which is precisely
the gap our lead-h protocol correction + DM-test-per-horizon contribution would fill.

---

### Synthesis: the 2026 evaluation standard, and what still isn't there (~430 words)

Read together, these eight works triangulate a 2026 review bar for epidemic-forecasting papers
that is considerably higher than what GNN papers were held to circa 2021–2023, but it is a bar
assembled from several *separate* demands, no single one of which any individual paper meets in
full. A reviewer in this space will now expect, at minimum: (1) a naive/seasonal or
persistence-style baseline reported at every horizon, framed as a genuine competitor rather than
a courtesy row — SpatialEpiBench, EpiCastBench, the Critical Audit, M-SPICE and "From naive to
foundation" all independently converge on the finding that simple baselines beat sophisticated
architectures on a large fraction of series/horizons, so a paper that doesn't show this comparison
now reads as incomplete rather than merely old-fashioned; (2) rolling-origin (prospective,
walk-forward) evaluation rather than a single train/test split, which five of the eight papers
now treat as table-stakes; (3) point *and* probabilistic scoring where feasible, with the
Forecast-Hub-standard WIS/relative-WIS/coverage machinery (Mathis et al., and by inference Cramer
et al.) increasingly cited as the canonical probabilistic protocol even by ML-venue benchmark
papers; (4) some form of formal multi-model significance testing — Friedman/Nemenyi/MCB
(EpiCastBench) is emerging as the ML-benchmark-paper norm, though it is still the exception, not
the rule; (5) explicit, auditable baseline and dataset provenance, following the Critical Audit's
demonstration that a benchmark's own data-loader (silent differencing) can manufacture GNN
"gains" that vanish on undifferenced data, and M-SPICE's admission that a standard baseline
(EARTH) simply would not run.

The gap none of the eight fills is exactly where our paper sits. **Not one of them computes a
minimum-detectable-effect or power analysis** for the comparisons it reports — every "model beats
baseline by X%" claim in this literature, including in a Nature Communications flagship
operational evaluation, is asserted without asking whether the test set is even large enough to
distinguish that effect from noise. **None reports Cola-GNN or EpiGNN broken out per-horizon
under a corrected scoring protocol** — the closest, SpatialEpiBench and M-SPICE, treat these as
one baseline among many in pooled multi-horizon tables, not as the object of a protocol-bug
investigation. And despite widespread awareness that foundation-model claims may rest on
contaminated or unrealistic benchmarks (Jafari et al. name this explicitly), no paper in the set
performs an actual pretraining-corpus contamination audit — awareness has outpaced verification
across the entire field. A paper that supplies a corrected lead-h protocol, a DM test with
power/MDE analysis, and a graph-free alternative would be read in 2026 as filling a real,
widely-acknowledged-but-unaddressed hole, not as redundant with any of these eight.

---

## Group 4: TERN + hybrid/generative epidemic-forecasting papers — close reading notes

**Reader note on access method.** TERN (paper 1) was read from the arXiv HTML rendering
(`arxiv.org/html/2609.18407v2`) via WebFetch, cross-checked directly against the actual source files of
`github.com/Neurogica/TERN` (README, `config/tern.json`, `src/train.py`, `src/models/tern.py`,
`src/utils/metrics.py`, `scripts/export_tables.py`, `tests/test_data.py`, `tests/conftest.py`, `.gitignore`,
file tree via `gh api`) — these code excerpts are byte-for-byte verbatim, fetched directly, not paraphrased.
Papers 2–6 were also read via WebFetch of their arXiv HTML pages; the returned text is WebFetch's own
summarization/extraction pass over the page, not something I paged through myself line by line, so quotes
below are reproduced as WebFetch returned them and should be treated as "quoted by the tool," not
independently re-verified against the PDF byte stream (the raw PDF fetch attempt failed — see below). I was
not able to open a clean full-text PDF or HTML for any of papers 2–6 through a second, independent route in
the time available, so anything below attributed to those five papers carries that one-hop-removed caveat.
Nothing is fabricated; where WebFetch's extraction was thin (e.g., some ablation tables, appendix content,
exact bibliographic pages/figure counts), I say so explicitly rather than padding it.

---

### 1. TERN — Nagashima & Funayama, arXiv 2609.18407v2 (24 Sep 2026)

#### A. Bibliographic + code
- Title: "TERN: A Delta-rule Memory with a Seasonal Reference and Online Adaptation for Epidemic Forecasting."
- Authors: Shunya Nagashima (Neurogica Inc.), Yuta Funayama (LTS, Inc.).
- v1: 16 Sep 2026; v2: 24 Sep 2026.
- Code: https://github.com/Neurogica/TERN, BSD-3-Clause-Clear license, `uv`-managed Python 3.10–3.12 /
  PyTorch 2.13 project. No affiliation/funding statement was surfaced by the HTML extraction.

#### B. Claims quoted verbatim (abstract, exact)
"Weekly influenza surveillance counts guide vaccine distribution and public-health alerts, yet they are hard
to forecast. Each region offers only a few seasons, waves shift in timing and height every year, and
information that helps while a wave grows misleads after its peak, whereas last season's shape stays
informative for a year. Existing epidemic graph models and general forecasters read a short fixed window and
treat all past information alike, so they neither exploit earlier seasons nor discard stale associations when
the epidemic phase changes. To address these limitations, we propose TERN, a forecaster built around a
delta-rule fast-weight memory that decays channel-wise and erases along a learned address under gates driven
by local epidemic-phase features, combined with an explicit seasonal reference and online adaptation. On
three Cola-GNN influenza benchmarks, TERN outperformed epidemic graph models and general forecasters, matched
or exceeded seasonal references, and a controlled comparison confirmed the contribution of the memory
itself."

Novelty claims (Introduction, per WebFetch extraction): "First delta-rule linear-attention forecaster for
epidemic surveillance"; a "controlled comparison isolating memory contribution" ("softmax attention degrades
RMSE on all three datasets" when context length, seasonal reference and online adaptation are held constant);
and "first side-by-side evaluation" of epidemic graph models, general forecasters, seasonal references and
zero-shot foundation models under one protocol, concluding "Tern is the best uncontaminated model on all
three influenza datasets."

#### C. Method (full detail, confirmed against `src/models/tern.py` source)
Per region, per head, one erase-then-delta fast-weight memory step (`step_eda` in code; matches paper Eq. 1–2):

```
S_t = (I − β_t k_t k_t^T) (I − γ_t e_t e_t^T) Diag(α_t) S_{t−1} + β_t k_t v_t^T
o_t = S_t^T q_t
```

- **Channel-wise decay**: α_t = exp(−r ⊙ m_t), r = softplus(ρ) learned per-channel rate, time constants
  τ = 1/r initialised log-uniformly on [1, 20] weeks (`l_max=20` = the protocol's window length); m_t =
  softplus(W_m z_t + b_m), initialised so m_t=1 at t=0 (zero weight init, bias = inv_softplus(1)).
- **Erase gate/address**: γ_t = σ(w_γ^T z_t) (bias initialised to −2, i.e. erase starts mostly off); address
  e_t = normalize(W_2 W_1 z_t) via a rank-16 factorisation (`erase_dim=16`) then a per-head (16×d_k) matrix.
- **Write (delta) gate**: β_t = σ(w_β^T z_t).
- q_t, k_t, v_t: linear projections → depthwise causal conv (kernel 4, "ShortConv", Mamba/GatedDeltaNet-style)
  → SiLU; q_t, k_t are ℓ2-normalised.
- **Phase-gating input**: z_t = [u_t; φ_t], φ_t = [Δx_t, Δ²x_t, Δx_t/(|x_{t−1}|+ε)] with ε=0.05, clipped to
  ±5 (growth rate, curvature, relative growth rate of the *normalised* series).
- **Region coupling**: one causal-attention "region attention" layer mixes regions after the per-region
  memory mixer (ablatable via `region_attention`; `adjacency_bias` optionally biases it with the dataset's
  adjacency matrix).
- **Seasonal reference**, two mechanisms: (i) a learned week-of-year embedding added to the token embedding
  (`season_embedding`), used in the Japan full-history config; (ii) an explicit climatology correction used
  for US-Regions/US-States: x̂_{t+h} = c_{t+h} + s_h · f_θ(x_{1:t}), c_{t+h} = mean over K seasons of the
  ±2-week window around t+h−52k (K = `climatology_seasons`, width = `climatology_width`), with shrinkage s_h
  per horizon (config values 0.5/0.3/0.1/0.05 for h=3/5/10/15 — i.e. the network's correction is trusted less
  and less at longer lead times, climatology dominates).
- **Online adaptation**, two mechanisms, both implemented in `src/train.py::train_full_history` and confirmed
  causal in the actual loop (see D below): (i) "online refit" — before scoring each test origin, take
  `--online_refit` gradient steps using only targets with `tgt <= t_now`; (ii) "online blend" — convex-combine
  the model forecast with the seasonal-naive x[t+h−52] using a weight α chosen by minimising loss over the
  last `online_blend` (default 12) origins whose targets are already observed at t_now (`online_alpha`,
  grid-searched over {0, 0.1, …, 1.0}).
- Ablation switches exposed as constructor args, confirming the paper's ablation axes exist as real code
  paths, not just descriptions: `rule` ∈ {eda (TERN), delta (Kimi Delta Attention/gated delta rule), gdn2
  (Gated DeltaNet-2), gla (gated linear attention, no delta correction)}; `decay` ∈ {channel, scalar, none};
  `mixer_type` ∈ {delta, attn (causal softmax attention swapped in)}.

#### D. Data & protocol — exact, confirmed against `src/data/...` test suite (see red flags: the loader module
itself is not visible on GitHub, but its unit tests are, and they pin down the exact behaviour)
- **Datasets**: three Cola-GNN influenza benchmarks — Japan-Prefectures (47 regions × 348 weeks, 2012–2019),
  US-Regions (10 HHS regions × 785 weeks, 2002–2017), US-States (49 states × 360 weeks, 2010–2017). Raw
  `.txt` count matrices (T×N) + adjacency `.txt` files, pulled by `scripts/clone_ext_repos.sh` from
  `amy-deng/colagnn`.
- **Split**: chronological 50/20/30 train/val/test, confirmed exactly by `tests/test_data.py`
  (`test_split_and_window_alignment`) on a synthetic 160-week/4-region series: `d.n_train, d.n_val == (80,
  112)`, i.e. train = rows 0–79 (50%), val = rows 80–111 (20%), test = rows 112–159 (30%) — matches the paper
  text "chronologically 50/20/30 into train/validation/test."
- **Normalisation**: per-region min–max, fit on training-period rows only. `tests/test_data.py::
  test_normalisation_uses_training_rows_only` explicitly asserts `d.max`/`d.min` are computed from
  `d.raw[train_idx[0]-window+1-h :][:window]` concatenated with `d.raw[train_idx]` — i.e. only rows drawn from
  the training span (including the lookback window immediately preceding the first training target, never
  rows from validation or test) are used for the min/max statistics. No leakage from val/test into
  normalisation stats, by direct test assertion.
- **Window/target construction**: input window = 20 steps ending at index i−h; target = x[i] (single-step,
  horizon h). Confirmed by `test_split_and_window_alignment`: `X[3] == d.dat[i-5+1-20 : i-5+1]`, `Y[3] ==
  d.dat[i]` for h=5. This is **lead-h-only scoring** — the model is retrained separately for each h ∈
  {3,5,10,15} (`--horizon` is a top-level CLI flag; `build_model`/`ColaGNNData` are re-instantiated with that
  h; `results/<dataset>/<tag>/h{h}_s{seed}.json` is one file per (dataset, tag, horizon, seed)). There is no
  pooling of predictions from a single multi-horizon model across leads — the protocol matches what the
  MSAGAT-Net paper-B correction argues is the fair way to score, not the "lead h..2h−1" baseline-scoring bug
  documented in `project_baseline_protocol_bug.md`.
- **Window vs. full-history regime**: window regime = fixed 20-week input, exactly the Cola-GNN/EpiGNN
  protocol, trained by minibatch SGD over sliding windows. Full-history regime (TERN only) reads the entire
  causal prefix x[:t] at every step and emits a forecast at every t via `data.stream(h)`, which returns
  `x_all` (the whole series) plus, per split, `(pos, tgt)` index pairs with `tgt = test_idx` and `pos = tgt −
  h`; confirmed causal by `test_stream_targets_match_window_protocol`. In `train_full_history`, the forward
  pass is literally `model(x_all[:, :upto], ...)` where `upto` is `n_train`/`n_val`/`data.n` for
  train/val/test evaluation respectively — full history "crossing into" validation/test *only in the sense
  that the causal prefix used to predict a test-period target legitimately includes earlier train+val rows*
  (which is standard for any sequence model with a lookback longer than the window); it never reads rows at
  or after the target index. This is legitimate, not leakage, and mirrors AGENTS.md's own note that "input
  windows may cross split boundaries (targets never do)" for the MSAGAT-Net LTLA/NHS data — TERN's
  full-history mode is the same idea taken to its logical extreme (unbounded lookback instead of a 20-step
  window).
- **Online adaptation causality**: `train_full_history`'s online-refit loop iterates test origins `t_now` in
  order and computes `observed = all_tgt <= t_now` before taking gradient steps or blending — i.e., at each
  test origin the model only ever sees targets whose true index is ≤ the current forecast origin. The paper
  text states this directly: "All of these operations use only information available at the origin." Code
  matches the claim; no look-ahead found.

#### E. Pooled RMSE definition — resolved precisely (from `src/utils/metrics.py` +
`scripts/export_tables.py`)
Two-stage pooling, confirmed by source, not just prose:
1. Within one (dataset, tag, seed, horizon) run, RMSE is **pooled over all test weeks and all regions** for
   that horizon: `rmse = sqrt(mean((y_pred − y_true)**2))` over the full (n_test_weeks × n_regions) array —
   this is `colagnn_metrics`'s `"rmse"` field (there is also an unused `"rmse_states"` variant that
   *averages* per-region RMSEs, matching an alternative metric some Cola-GNN-lineage papers report, but the
   main tables use pooled `"rmse"`).
2. Across horizons, `scripts/export_tables.py::horizon_average` groups by horizon, takes the **mean (or,
   for the EpiGNN re-run, median) over seeds first**, then takes the **simple arithmetic mean of the four
   per-horizon RMSE numbers** — i.e. "pooled RMSE" in Table 1 is *the average of four already-pooled-within-
   horizon RMSEs*, not a single RMSE computed by pooling squared errors across all four horizons together.
   This matters: it means the headline number is not driven by whichever horizon has the most test rows;
   each horizon counts equally regardless of how many (region, week) pairs it contributed.
- Each horizon is genuinely lead-h-only (per D above) — there is no "score horizon h using a model trained at
  a different horizon and reindexed" trick anywhere in the pipeline.

#### F. Baselines — re-run vs. copied
- **Copied from cited papers** (not re-run): Cola-GNN, EpiGNN "published" numbers (loaded from
  `data/published/epignn_table2.json` per `export_tables.py`, i.e. transcribed from Table 2 of the EpiGNN
  paper, not reproduced).
- **Re-run by TERN's authors**, same split/hyperparameter protocol (Adam 1e-3, weight decay 5e-4, batch 128,
  ≤1500 epochs, 5 seeds): EpiGNN (official code, seed-median reported "as some seeds diverge"); DLinear,
  PatchTST, iTransformer, TimeMixer, TimesNet, all via `thuml/Time-Series-Library` wrapped in
  `src/models/tslib.py` (d_model=64, 2 layers).
- **Zero-shot, not retrained**: Chronos-2 and TimesFM-3, run in both 20-week-window and full-history context
  via `scripts/zero_shot.py` (needs `chronos-forecasting` pip package).
- **Naive references**, author-computed: seasonal-naive (52-week lag) and 2-season climatology
  (`scripts/naive_baselines.py`).

#### G. Headline numbers (Table 1, pooled RMSE↓ / PCC↑, averaged over h∈{3,5,10,15}, 5 seeds)

| Method | Japan-Pref. RMSE/PCC | US-Regions RMSE/PCC | US-States RMSE/PCC |
|---|---|---|---|
| Seasonal naive (52wk) | 839 / 0.913 | 876 / 0.802 | 307 / 0.764 |
| Climatology (2 seasons) | 1030 / 0.876 | 727 / 0.862 | 265 / 0.812 |
| Cola-GNN (reported) | 1254 / 0.839 | 957 / 0.775 | 212 / 0.877 |
| EpiGNN (reported) | 1234 / 0.831 | 852 / 0.799 | 200 / 0.892 |
| EpiGNN (re-run, seed median) | 1379 / 0.765 | 961 / 0.713 | 217 / 0.872 |
| DLinear / PatchTST / iTransformer / TimeMixer / TimesNet | 1731–2039 / 0.185–0.543 | 991–1109 / 0.599–0.695 | 220–288 / 0.776–0.865 |
| Chronos-2 (zero-shot)† | 1314 / 0.776 | 698 / 0.877 | 207 / 0.883 |
| TimesFM-3 (zero-shot)† | 1498 / 0.683 | 340 / 0.972 | 139 / 0.949 |
| **TERN (window)** | 1145 / 0.857 | 885 / 0.775 | 206 / 0.885 |
| **TERN (full-history)** | **838 / 0.926** | **698 / 0.872** | **199 / 0.898** |

†marked as pretraining-contaminated on US datasets (see I below), "shown, not ranked."

Per-horizon breakdown was not given as a full table in the HTML extraction (only qualitative statements): TERN's
gains "concentrate at short lead times" — e.g. on US-Regions "20% below climatology at h=3, but 5.5% above at
h=10"; on Japan "2.9% above seasonal naive at h=5"; TERN beats seasonal naive "in 64–77% of prefectures at
every lead time" and beats climatology "in 67–96% of US states, but only at h=3 (10/10) and h=15 (6/10) on
HHS regions." I could not locate an actual per-region-and-horizon numeric table in what WebFetch returned;
this may exist only in a figure/appendix not captured by the HTML extraction pass.

#### H. Ablations (Table 2, full-history regime, seeds 0–2 only, averaged over 4 horizons)
Thirteen single-component swaps (GDN-2 rule, no erase/KDA, no decay, no phase features, no region attention,
softmax mixer, no seasonal reference, constant shrinkage s_h, unweighted loss, no online blend, no online
refit, no Polyak averaging, no adjacency bias), each run against the full TERN configuration. Full TERN wins
or ties on RMSE in every row shown; deltas are mostly small (single digits to ~15%) except "no seasonal
reference" on US-Regions (941 vs. 691, a ~36% RMSE increase) and "no decay" on US-States (250 vs. 195, a ~28%
increase) — the two largest single-component effects. Notably: **ablations use only 3 seeds, not the 5 used
for the headline table**, and report point estimates with no confidence intervals or significance tests.

#### I. Chronos-2 / TimesFM-3 contamination claim
Exact text (as extracted): "The GiftEvalPretrain corpus [24] used by TimesFM-3 and Chronos-2 contains CDC
FluView ILINet and WHO/NREVSS series underlying US-Regions and US-States. The RMSE of TimesFM-3 on
US-Regions barely depends on lead time (318–355), unlike Japanese data not in corpus." Table 1 marks the two
zero-shot models on US-Regions/US-States with a dagger footnoted "Pretraining corpus contains CDC FluView
series of US datasets (shown, not ranked)," and `scripts/export_tables.py` hard-codes this via a
`CONTAMINATED = ("region785", "state360")` constant that suppresses ranking (bold/underline) for those two
rows on those two datasets — i.e. this is a real, code-level policy decision by the authors, not just a
footnote. The evidence offered is indirect (a named external corpus said to contain the source series, plus
the observed insensitivity of TimesFM-3's RMSE to lead time on US-Regions as circumstantial support) rather
than a direct audit of Chronos-2/TimesFM-3's training manifests; I could not verify GiftEvalPretrain's actual
contents independently in this session.

#### J. Compute cost
"Full-history run takes ~40 minutes on one NVIDIA RTX PRO 6000 GPU, window baselines take seconds." Model
sizes: window regime d_model=64 (US-States dropout 0.4) → 125k params; full-history regime d_model=32 (dropout
0.5; US-States d_model=64, dropout 0.4) → 40k params. Both regimes are small models by modern standards.

#### Seeds, variance, significance
Five seeds {0,1,2,3,4} for the main table (default `make_jobs.py --seeds 0,1,2,3,4`); ablations use seeds
{0,1,2} only. EpiGNN re-run reports the seed **median** per lead time rather than mean "as some seeds
diverge" — an explicit acknowledgment of training instability for that baseline. **No significance testing of
any kind** (no DM test, no bootstrap CI, no paired test) appears anywhere in the paper or the table-export
code; results are marked "best bold, second underlined" by literal numeric comparison at the displayed
precision (`mark()` in `export_tables.py`), which is exactly the kind of "who's bolded" ranking your target
paper's power-analysis/pooled-DM-test framing is positioned to critique.

#### Limitations (verbatim, per WebFetch extraction)
"The main limitation is the scope of the evaluation, which is confined to influenza surveillance on three
benchmarks. In future work, we plan to add probabilistic outputs and to cover other diseases and
hospitalisation targets." This is a short, single-paragraph limitations statement — no discussion of the
missing significance testing, no discussion of graph-attention/region-attention interpretability, no mention
of the compute asymmetry between full-history TERN and the "seconds"-scale baselines when judging fairness of
comparison.

#### Red flags
1. **The data-loading/split module (`src/data/colagnn.py`, referenced by name in the README and imported by
   `src/train.py` as `from data import ColaGNNData`) is not present in the public GitHub tree.** I confirmed
   this via `gh api repos/Neurogica/TERN/git/trees/main?recursive=1` (not truncated) — no `src/data` path
   exists anywhere in the repo. The cause is visible in `.gitignore`: the line `data/` (no leading slash, no
   trailing `/py` restriction) matches *any* directory literally named `data` at *any* depth, including
   `src/data/`, not just the top-level `data/` (raw dataset cache) it was presumably meant to exclude. The
   result: the single most decision-relevant piece of code for verifying the split/normalisation protocol —
   the actual `ColaGNNData` class — is silently absent from the repository as currently published, even
   though the README instructs users to look at "the loader in `src/data/colagnn.py`." I was only able to
   reconstruct its exact behaviour indirectly, via `tests/test_data.py` and `tests/conftest.py`, which import
   and exercise it and whose assertions (quoted in D above) are strong but not a substitute for reading the
   implementation itself — e.g. I cannot verify how `ColaGNNData.stream()` builds `x_all` for datasets with
   missing weeks, or exactly how per-region weighting interacts with regions that have zero variance in the
   training window. **This should be flagged as a reproducibility gap if TERN is cited or compared against**:
   as published, `git clone` + `uv sync` + the documented commands will fail with `ModuleNotFoundError: No
   module named 'data'`.
2. Ablations use 3 seeds while headline results use 5, with no stated justification (likely just compute
   cost) — makes the ablation numbers less trustworthy for close comparisons, and the paper doesn't flag this
   asymmetry itself.
3. No significance testing anywhere, despite ranking by whichever number is numerically lower to two/three
   significant figures — several "best" vs "second" marks in Table 1/2 are separated by <2% RMSE, well within
   plausible seed noise given only 3–5 seeds and no reported spread.
4. The contamination-adjustment for Chronos-2/TimesFM-3 is a good-faith, transparent move (kudos — it's coded
   as a policy, not just prose), but it is not accompanied by any attempt to test whether TERN's *own* full-
   history regime, which also reads the entire causal history, might benefit asymmetrically on datasets with
   longer histories (US-Regions has 785 weeks vs. Japan's 348) — the paper doesn't discuss whether "full
   history" advantages scale with series length in a way that could itself be a confound.
5. Single-paper self-comparison: no external significance test against baselines, no confidence interval on
   the headline pooled RMSE, so "TERN is the best uncontaminated model" is asserted from point estimates only.
6. The paper's "full-history" and "window" regimes use *different hyperparameters per dataset* (different
   d_model, dropout, and enabled features — e.g. `adjacency_bias` only for US-States, `season_embedding` only
   for Japan, climatology only for US-Regions) chosen presumably by validation performance; this is reasonable
   ML practice but means the "one model, one architecture" framing in the abstract undersells how much
   per-dataset architecture search went into the final config (visible directly in `config/tern.json`).

#### How to run TERN under our protocol (chronological 60/20/20, window 20, horizons 3/5/10/15 weekly and
3/7/14 daily, lead-h scoring only, 5 seeds 42/30/45/123/1000, train-range-only normalisation)

**What already matches, unchanged:**
- Lead-h-only scoring: TERN already retrains per horizon (`--horizon` flag), never scores a horizon using a
  different lead's forecast. This matches our protocol directly — no change needed to the scoring semantics.
- Window length 20: TERN's default `--window 20` is already our value for the weekly datasets.
- Train-only normalisation: `ColaGNNData`'s min-max-from-training-rows behaviour (as pinned by
  `tests/test_data.py`) matches our "normalisation fit on the training range only" rule in AGENTS.md §4 — no
  change needed in principle, *if* the missing loader is reconstructed faithfully (see below).
- Seeds: TERN's `--seed` is a plain int fed to `set_seed()` (Python/NumPy/Torch); our seed set
  {42,30,45,123,1000} can be passed directly with no code change, just `make_jobs.py --seeds 42,30,45,123,1000`
  (or bypass `make_jobs.py`/`run_queue.py` and call `src/train.py` five times).

**What must change:**
1. **Split fraction, 50/20/30 → 60/20/20.** This is hard-coded inside the missing `ColaGNNData.__init__`
   (visible only through its externally-observed behaviour: `n_train = 0.5 * T`, `n_val = 0.7 * T` rounding as
   in the synthetic test). Since the source file isn't in the repo, this cannot be patched by flag — it would
   need to be reconstructed from the test-pinned behaviour and then edited (change the split fractions), or
   the loader would need to be rewritten from scratch against our own `DataBasicLoader`
   (`src/data.py` in MSAGAT-Net) semantics. This is the single largest blocker to "just running" TERN
   unmodified: **the file we'd need to edit does not exist in the public repo.**
2. **New datasets (NHS 7 regions, LTLA 372 regions, Australia 8 states, daily).** TERN's loader is keyed by a
   `DATASETS` dict mapping a dataset name to `(counts_file, adjacency_file, n_regions)` (per
   `tests/test_data.py`: `colagnn.DATASETS["japan"] = ("japan.txt", "japan-adj.txt", 4)`), reading plain
   whitespace/comma-delimited T×N count matrices plus an N×N adjacency matrix. Our data lives as CSVs in
   `data/` with different column/date conventions; porting means either (a) writing an export step that dumps
   our per-dataset count matrices and existing adjacency matrices (already present per AGENTS.md §2 — NHS,
   LTLA and Australia geometries) into Cola-GNN's flat-text format and registering them in the `DATASETS`
   dict, or (b) rewriting the loader against our own `DataBasicLoader`. Either way requires the missing
   `colagnn.py` file to exist first.
3. **Daily horizons and the 52-week assumptions.** Several defaults are annual-cycle-specific to *weekly*
   data: `--blend_period` defaults to 52 (the seasonal-naive lag, in steps); the climatology correction's
   `climatology_seasons`/`climatology_width` and the week-of-year `season_embedding` are implicitly indexed
   mod-52. For our daily panels (NHS/LTLA/Australia) and horizons 3/7/14 days, every one of these would need
   to be re-parameterised to a ~365-day cycle (`--blend_period 365`, climatology width in days not weeks, and
   the season-embedding index would need to be computed mod-365/366 rather than mod-52 — I could not confirm
   from the visible code whether the season index is derived from a `week_of_year` column that is itself
   hard-coded, or generically from `t mod period`; if the latter, only a CLI-level `--period` argument would
   be needed, but neither `train.py`'s argparse block nor the (missing) loader expose such a flag today, so
   this is at minimum a small code change, not just a flag change).
4. **Online-adaptation window sizing.** `--online_blend 12` (12 origins) and `--online_refit` step counts were
   tuned for weekly cadence; for daily data with 372 regions, both the "last 12 origins" window and the
   per-origin gradient-step cost would need re-tuning (more origins for a meaningful rolling-error estimate at
   daily granularity, and 372-region full-history forward passes are more expensive per step than 47/49/10).
5. **Region attention / adjacency bias at N=372.** TERN's region-coupling layer is one causal-attention layer
   over regions (`region_attention`, optionally `adjacency_bias`-biased). The datasets TERN was tested on cap
   out at 49 regions; LTLA has 372. This is architecturally fine (attention over 372 regions per timestep is
   not prohibitive), but it is untested territory for TERN and its authors offer no comment on region-count
   scaling. The 40k–125k parameter counts reported (§J) should scale gently with N through the linear
   input/output projections, but the region-attention layer's cost is O(N²) per timestep in that layer alone
   (same caveat AGENTS.md/ledger E13 raises for MSAGAT-Net's own spatial module) — worth flagging as a
   comparability caution, not a blocker.
6. **What cannot be made comparable, even after all edits.** TERN's zero-shot foundation-model baselines
   (Chronos-2, TimesFM-3) were flagged contaminated specifically because GiftEvalPretrain is alleged to
   contain the *named* Cola-GNN US series; whether it also contains NHS/LTLA/Australia COVID panels is
   unknown and would need the same due-diligence TERN's authors did for their own datasets before any
   zero-shot foundation-model row could be trusted in a comparison against our data. Likewise, TERN's
   "full-history" climatology term is calibrated on two-season week-of-year statistics for influenza series
   that have a clean single-peak annual cycle; COVID incidence (multi-wave, irregular seasonality, especially
   in the earlier pandemic years) may not reward the same climatology-shrinkage structure, so even a faithful
   port might show a smaller or no full-history advantage — that would be a legitimate empirical finding to
   report, not a protocol violation.

**Net assessment:** TERN's *training-loop* logic (window/full-history split, online-adaptation causality,
lead-h scoring, per-horizon retraining) is directly reusable and already close to a "fair" protocol by our
own standards. The blocking obstacle is that the actual data-loading/normalisation module is missing from the
published repository (item 1 above), so faithfully reproducing even the existing three benchmarks under our
own protocol requires either waiting for the authors to fix `.gitignore` and push `src/data/colagnn.py`, or
reverse-engineering it from the pinned test behaviour before any of the split/porting work in items 2–5 can
begin.

---

### 2. Su, Lee, Cui, Ramakrishnan — "How (Not) to Hybridize Neural and Mechanistic Models for Epidemiological
Forecasting," arXiv 2602.06323

#### A. Bibliographic
Authors Yiqi Su, Ray Lee, Jiaming Cui, Naren Ramakrishnan. v1 6 Feb 2026, v3 18 Aug 2026. cs.LG, CC BY 4.0.
Proposed model is internally named **EpiNode** in the WebFetch extraction (not stated in the given title/
abstract, but this is the name used throughout the method section as returned). Code link was not surfaced by
the extraction; I did not locate a GitHub repository for this paper in this session and cannot confirm one
exists.

#### B. Claims (quoted as extracted)
Central methodological framing: decompose observed infections I(t) = T(t) + S(t) + R(t) (trend/seasonal/
residual, via Variational Mode Decomposition) and use these three signals as *interpretable control inputs*
to three coupled latent neural ODEs that jointly decode time-varying SIRS parameters (β, γ, δ), rather than
letting a single neural ODE freely drive the mechanistic model end to end.

#### C. Method (5–10 lines)
Three "collaborative" latent neural ODEs, one per decomposed component (trend/seasonal/residual), evolve
independently and fuse into a shared representation that decodes bounded, interpretable epidemiological
parameters, e.g. β(t) = β_min + (β_max − β_min)·β̃(t), β̃∈(0,1), which then drive a mechanistic SIRS
compartmental system forward. The paper frames this as a fix for four named failure modes of naive
neural-mechanistic hybrids (see G/novelty below): (1) neural ODEs fail at long-horizon forecasting even with
full state observability and mass-conservation constraints; (2) bidirectional training objectives don't
resolve identifiability or learn plausible latents under partial observability; (3) physics-informed losses
degrade on partially-observed, complex dynamics (SIRS with I-only) versus simple ones (SIR); (4) mechanistic
neural ODEs can't capture real multi-wave dynamics because latent forcing (seasonality, immunity waning) is
unidentified.

#### D. Data & protocol
Synthetic: SIRS (fixed and time-varying parameters), SIR, SEIRS, all simulated from known compartmental
models with ground-truth parameters for validation. Real-world: weekly US CDC ILI surveillance, 10 HHS
regions, week 30 2022–week 30 2025. Splits are dataset-specific ratios reported per dataset (e.g. 0.3/0.7 for
SIRS-Fixed, 0.7/0.3 for ILI) — **not a uniform chronological 60/20/20**, and it's unclear from the extraction
whether these splits are chronological at all (plausible for time-series but not explicitly confirmed by the
text WebFetch returned). No mention of a held-out validation split distinct from train/test in what was
extracted — this may mean the "train/test" language in the summary elides a validation split used for
early-stopping/model selection, or that the paper genuinely reports only two-way splits; I could not confirm
which from the extraction.

#### E. Baselines
ARIMA, LSTM, EINN (adapted SEIRm→SIRS), NeuralODE, LatentODE, KAN-ODEs, EARTH, TimeKAN, TimeMixer++. Whether
these were re-run under identical conditions or partly copied from their original papers was not stated
explicitly in the extraction; standard deviations are reported for EpiNode's own results, suggesting multiple
seeds, but the number of seeds was not given.

#### F. Headline numbers
RMSE on infection trajectories, EpiNode best on all five datasets in the extraction: SIRS-Fixed 0.0022 vs.
0.0046 (LatentODE); SIRS-Varying 0.0195 vs. 0.0405 (LatentODE); ILI 0.0093 vs. 0.0209 (KAN-ODEs). Peak-timing
bias 0.0 weeks (σ=0.7). Parameter-recovery sign agreement for β(t): 66.7% vs. EINN-SIRS's 59.8%.

#### G. Ablations
Single vs. three collaborative latent ODEs; 1/2/3-component decomposition; decomposition method (moving
average, STL, VMD, Wavelet, SSA-VMD, Neural Koopman); time-delay embedding on/off. VMD with three components
plus time-delay embedding wins.

#### H. How written
Structured around explicitly named "failure modes" (a diagnostic Section 2) before presenting the fix — a
"here's what breaks, here's why, here's the fix" argument structure, which is a stronger novelty framing than
a pure leaderboard paper. Uses both fully synthetic (known ground truth) and real ILI data, which lets it
make identifiability/parameter-recovery claims that pure real-data papers cannot. Limitations section is
present and reasonably candid (see below).

#### I. Red flags
- Non-uniform, dataset-specific train/test ratios reduce comparability across the five datasets within the
  paper's own tables. Whether splits are chronological was not confirmed.
- No explicit mention of significance testing (DM test, CIs) in what was extracted.
- Single-region ILI aggregation at 10-HHS-region granularity is coarse relative to LTLA/NHS-scale evaluation;
  no spatial coupling is modeled (explicitly listed as a limitation by the authors themselves).

#### Limitations (as extracted)
Single-region, deterministic dynamics, no explicit uncertainty quantification; no spatial coupling or
intervention modeling; TSR decomposition is a fixed preprocessing step, not learned; performance depends on
decomposition choice; future work: probabilistic extensions, spatial interactions, learnable decomposition.

#### J. Relevance to the target paper (i)–(v)
- (i) lead-h protocol correction: not directly relevant — this paper does not appear to make the
  lead-h-vs-pooled-horizon evaluation mistake an issue at all, since it's mechanistic-hybrid focused, but its
  non-uniform splits are a different-but-related evaluation-fairness weakness worth citing as another example
  of "protocol details silently vary across papers in this literature."
  (ii) graph attention collapse: not applicable — no graph/spatial attention module exists in this method
  (explicitly listed as a limitation: "no spatial coupling"). Could be cited as a contrast: a model that
  *avoids* spatial machinery entirely and is honest about it, versus MSAGAT-Net's spatial attention that is
  present but inert.
  (iii) scale-equivariant, graph-free common-factor model: this paper's use of trend/seasonal/residual
  decomposition as an *explicit, interpretable control signal* is conceptually adjacent to a seasonal-memory
  argument, and could be cited as prior evidence that seasonal decomposition helps when done explicitly rather
  than left for attention to discover — supports your architectural instinct.
  (iv) power analysis/DM tests/conformal intervals: not present in this paper as far as extracted; a gap you
  can point to.
  (v) COVID daily panels at NHS/LTLA/Australia scale: not evaluated here (weekly HHS-region ILI only, single
  country) — no direct overlap, but the four "failure modes" taxonomy is a useful citation for the "neural
  hybrids fail in specific, nameable ways" argument regardless of dataset overlap.

---

### 3. Komodromos, Malialis, Kontou, Kolios — "Cross-Country Learning for National Infectious Disease
Forecasting," arXiv 2601.20771 (EMBC 2026)

#### A. Bibliographic
v1 28 Jan 2026, v3 17 Sep 2026 (latest). q-bio.PE / cs.LG. 7 pages, 4 figures, 5 tables. IEEE EMBC 2026
(Toronto) — a conference paper, not a journal article; short-format (7pp) is consistent with an EMBC
extended-abstract-style contribution rather than a full archival paper. No code link surfaced.

#### B. Claims (quoted)
"Accurate forecasting of infectious disease incidence is critical for public health planning and timely
intervention. While most data-driven forecasting approaches rely primarily on historical data from a single
country, such data are often limited in length and variability, restricting the performance of machine
learning (ML) models." Core proposal: pool time series across 46 European countries to train one model,
evaluated on Cyprus.

#### C. Method
Single model trained on pooled multi-country input–output window pairs (lookback 7/14/21 days tested, 14 best;
horizon fixed at 7 days ahead), constructed identically across countries; compared against nation-only
training. Models: naive-last-value, last-week-average, seasonal-naive, ARIMA (AIC-selected order), XGBoost,
Transformer.

#### D. Data & protocol
46 European countries + Cyprus target, 2020-01-01–2022-12-31, Oxford COVID-19 Government Response Tracker
(OxCGRT) supplemented by Cyprus Ministry of Health weekly reports. Countries with >1/9 missing reporting days
excluded. Missing days interpolated by evenly distributing weekly totals across 7 days (a real preprocessing
choice worth flagging — this smooths/attenuates day-of-week structure and could itself bias short-horizon
daily forecasts). log1p transform; per-country z-score standardisation for neural models. **Three splits**,
none of which is a simple chronological 60/20/20: Split 1 = first half of the largest wave for train, remainder
+ final wave for test; Split 2 = three smaller waves for train, low-activity periods for test; Split 3 =
later-stage data for both train and test. This is a deliberately *regime-shift* evaluation design (train on
one epidemic phase, test on a different one) rather than a proportional holdout, which is a different kind of
protocol rigor from what our target paper is arguing for, but conceptually related: both are pushing back on
naive "just hold out the last N%" splitting.

#### E. Baselines
Naive/seasonal-naive/last-week-average/ARIMA are simple, reproducible references; XGBoost and Transformer are
re-run by the authors (15 repetitions each, with SDs reported) — i.e. genuinely re-run, not copied.

#### F. Headline numbers
14-day lookback best across models. Split 1: XGBoost all-countries MAE 2029 (18.7% MAPE) vs. national-only MAE
2671 (21.4% MAPE). Split 2: XGBoost all-countries MAPE 19.2% vs. ARIMA 27.6%. Cross-country pooling
consistently helps across all three splits.

#### G. Ablations
Country-selection ablation: national-only vs. correlated (|ρ|≥0.3 with Cyprus) vs. uncorrelated (|ρ|<0.3) vs.
all-countries. All-countries wins even over "correlated-only," suggesting information at different temporal
phases from ostensibly "uncorrelated" countries still helps — a genuinely interesting negative result against
naive similarity-based country selection.

#### H. How written
Short (7pp) EMBC-format paper: compact method, three real train/test regime splits used as the main
experimental axis (a nice design choice for demonstrating generalization robustness under distribution shift,
appropriate for the paper's claimed contribution), five results tables. Limited room for appendix depth (per
the WebFetch extraction, no explicit appendix content was found — consistent with EMBC's page limit).

#### I. Red flags
- Single target country (Cyprus) — external validity of "cross-country pooling helps" rests on one downstream
  country; the paper does not claim (per extraction) to test the same pooling with a different target country
  to check the result isn't Cyprus-specific.
- No mention of a held-out validation split distinct from train/test for hyperparameter tuning — risk of
  implicit test-set peeking during lookback-window and model selection, though 15-repetition SD reporting
  suggests some rigor.
- No significance test connecting the reported SDs to the point differences claimed (e.g. is 18.7% vs 21.4%
  MAPE significant given repetition variance?) — extraction did not surface one.

#### J. Relevance to target paper (i)–(v)
- (i) lead-h protocol: horizon is fixed at 7 days single-step, no multi-horizon pooling issue to critique here
  — not directly relevant to your protocol-correction argument, but its three non-proportional, regime-shift
  splits are a useful supporting citation for "evaluation protocol choices matter and vary a lot across this
  literature."
  (ii) graph attention collapse: not applicable, no spatial/graph model — this paper's cross-country
  "transfer" is a pooled-training strategy, not an attention mechanism, so no direct overlap.
  (iii) graph-free common-factor model: mildly relevant as a precedent — this paper's core finding is that
  *pooling across the population without explicit graph structure* (no adjacency, no distance weighting)
  outperforms restricting to "correlated" countries, an indirect data point supporting the idea that explicit
  graph/attention machinery may add less than pooling/shared statistical strength does.
  (iv) power analysis / DM / conformal: not present here.
  (v) daily COVID panels, test-levels-exceed-validation: Split 1's design (train on first half of largest
  wave, test on remainder + a later wave) plausibly puts higher incidence in the test set than in at least
  part of training, similar in spirit to your NHS/LTLA/Australia "test exceeds validation" framing — worth a
  citation for "others have also had to design around regime shift between train and test incidence levels,"
  though the specific mechanism (deliberately chosen wave-based splits vs. chronologically-inherited
  extrapolation) differs.

---

### 4. Pathak & Chakraborty — "Deep Generative Spatiotemporal Engression for Probabilistic Forecasting of
Epidemics," arXiv 2603.07108 (TMLR 2026)

#### A. Bibliographic
v1 7 Mar 2026, v2 6 Jul 2026. Published/accepted at Transactions on Machine Learning Research (TMLR), 2026 —
a peer-reviewed journal-track venue (this is a meaningfully stronger venue signal than most of the other
papers in this group, which are arXiv preprints or workshop/conference papers). stat.ML/cs.LG/stat.ME. Code:
GitHub repository plus a PyPI package `stengression` — i.e. this is the one paper in the group besides TERN
with a publicly distributed, installable software artifact, which is a positive reproducibility signal (I did
not independently inspect the repo's contents in this session).

#### B. Claims (quoted)
"Probabilistic forecasting of epidemics is therefore crucial for providing the best or worst-case scenarios
rather than a simple, often inaccurate, point estimate." Core technical claim: a **pre-additive noise**
generative structure, y_t = f(y_{t-1} + η_t) rather than post-additive (noise added after the nonlinearity),
lets the network act as a "distributional lens" propagating stochasticity through nonlinear layers, trained
with an energy-score loss so the predicted distribution converges to the true one rather than exhibiting the
uncertainty inflation typical of models like DeepAR.

#### C. Method
Three architectures sharing the pre-additive-noise + energy-score-loss recipe: GCEN (GCN spatial embedding +
LSTM), STEN (STAR-layer distance-weighted spatial lags + LSTM), MVEN (pure per-node LSTM, no spatial term, a
built-in ablation/baseline). Uncertainty comes from repeating the forward pass M times with resampled noise to
build an ensemble of trajectories; prediction intervals are ensemble quantiles, no external calibration step
needed by construction. A theoretical result (Theorem 2) establishes geometric ergodicity/asymptotic
stationarity for GCEN/STEN under stated assumptions — i.e. the influence of initial conditions decays
exponentially, a formal stability guarantee rarely offered in this literature.

#### D. Data & protocol
Six datasets: Japan TB (47 regions, monthly, 1998–2015), China TB (31 regions, monthly, 2014–2018), USA ILI
(50 regions, weekly, Oct 2010–Jan 2017 — note: overlaps in spirit with the Cola-GNN US-States series TERN
uses, though the extraction did not confirm whether it's literally the same underlying data), Belgium COVID
(11 regions, daily, Sep 2020–Oct 2022), Colombia Dengue (33 regions, weekly, 2007–2022), Hungary Varicella (20
regions, weekly, 2005–2014). "A strict temporal split is employed... where the final segment of each time
series corresponding to the forecast horizon is held out as the test set," with validation-set length details
deferred to Appendix E.1 (not extracted here — I did not fetch the appendix directly). Per-node
standardisation (mean/SD, not train-only explicitly confirmed by the extraction, though "strict temporal
split" language suggests it is). Multiple forecast horizons per frequency: monthly {6,12,24}, daily
{30,60,90}, weekly {4,9,13}.

#### E. Baselines
Twelve models: LSTM, NHiTS, Transformers, TCN (temporal); GSTAR, STARMA, STGCN (spatiotemporal-traditional);
DeepAR, Prob-iTransformer, GpGp, DiffSTG, STESN (probabilistic). Stated as "reimplemented" by the authors
(implementation specs in Appendix E) — i.e. re-run, not copied, for all twelve.

#### F. Headline numbers
Point-forecast metrics (SMAPE, MAE, RMSE, MASE, RMSSE) and probabilistic metrics (Pinball-80/95, ρ-risk at
ρ=0.5/0.9, CRPS, Winkler score) are both reported; the extraction returned only the qualitative summary
("either outperform the benchmarks, or remain highly competitive") with actual numeric tables deferred to
Appendix G, which I was not able to pull into this pass — **numeric headline results for this paper are not
independently confirmed in these notes; only the metric list and qualitative claim are**.

#### G. Ablations
Section 8: pre-additive vs. post-additive noise; energy-score loss vs. MSE training; spatial-module ablation
(GCEN vs. STEN vs. MVEN, i.e. graph-conv spatial term vs. distance-weighted spatial term vs. no spatial term
at all) — this last ablation axis is directly relevant to the target paper's "graph attention collapses to
uniform pooling" argument, since MVEN is effectively the "graph-free" arm of their own ablation.

#### H. How written
TMLR-track paper with a formal theory section (geometric ergodicity proof) alongside six empirical datasets
and both point and probabilistic metrics — the most methodologically dense/rigorous paper in this group by
structure (theory + six datasets + both point/probabilistic eval + reimplemented baselines + released
package). Explicit appendices (E for baseline specs, G for full result tables) that I did not fully retrieve.

#### I. Red flags
- I could not verify the actual headline numbers (only the qualitative "competitive or better" framing) —
  this is a genuine gap in my read, not a criticism of the paper; flagging for whoever synthesizes across
  groups that this paper's specific win-margins need a follow-up read of Appendix G before citing exact
  numbers.
- Validation-set length is deferred to an appendix not retrieved here.
- "Framework currently restricted to single-target forecasting (D=1)" (self-reported limitation) — cannot
  jointly model multiple observed channels (e.g. cases + deaths) per region.

#### Limitations (as extracted)
Single-target only (D=1); theoretical analysis uses a simplified closed-loop setup without exogenous
covariates; STEN's compute scales with data size; practical implementations omit the hidden-state noise
injection used in the proofs (a theory-practice gap the authors flag themselves). Future work: multi-target
forecasting, multivariate epidemic indicators, exogenous covariates (mobility, weather).

#### J. Relevance to target paper (i)–(v)
(i) lead-h protocol: uses multiple discrete horizons per frequency (not a single pooled multi-horizon score,
per the "strict temporal split... final segment... held out" language, though whether each horizon is scored
separately or pooled together like TERN's approach was not confirmed) — worth a direct follow-up read of the
results section to see whether this paper makes the same pooled-horizon mistake your paper is correcting, or
avoids it.
(ii) graph attention collapse: the MVEN-vs-GCEN/STEN ablation (spatial module completely removed) is a
directly citable structural precedent for testing whether spatial structure earns its keep — if MVEN
(no spatial term) is competitive with GCEN/STEN in their own Section 8 ablation, that would be independent
supporting evidence for your claim that graph machinery often isn't pulling its weight in this literature; I
was not able to confirm the actual ablation *numbers* in this pass, only that the axis exists.
(iii) scale-equivariant graph-free common-factor model with seasonal memory: this paper's per-node LSTM (MVEN)
baseline, its energy-score/pre-additive-noise machinery, and geometric-ergodicity theory are all complementary
territory — a probabilistic, graph-free architecture with theoretical stationarity guarantees is a reasonable
thing to contrast your horizon-monotone gate model against, especially on the probabilistic-forecasting axis.
(iv) power analysis/DM/conformal: this paper reports CRPS, Winkler score and pinball loss (calibration-style
probabilistic metrics) but nothing in the extraction suggests DM tests or conformal intervals specifically —
its ensemble-quantile intervals are a *different* uncertainty-quantification route than conformal, worth
contrasting explicitly (ensemble-based generative UQ vs. distribution-free conformal UQ).
(v) daily COVID panels at NHS/LTLA/Australia scale: the closest overlap is Belgium COVID (11 regions, daily) —
a genuinely relevant same-disease, same-frequency comparison point, though at much coarser regional
granularity (11 vs. 372) than LTLA.

---

### 5. Hua & Bu — "GeoID-PINN: Identifiability-Aware Regional Epidemic Inference with Geographic Coupling,"
arXiv 2608.02633 (epiDAMIK @ KDD 2026)

#### A. Bibliographic
v1 28 Jul 2026. cs.LG/stat.AP/stat.ML. Accepted, 8th epiDAMIK ACM SIGKDD Workshop (KDD 2026), 11 pages, 3
figures — a workshop paper, shorter/less mature venue than a full conference/journal track. No code link
surfaced.

#### B. Claims (as extracted)
Introduces a physics-informed neural network for SIRD dynamics coupled across regions via a "row-stochastic
source-composition matrix" incorporating distance, adjacency, commuting or lead-lag-inferred priors, with an
explicit identifiability-regularisation term toward that spatial prior. Central methodological warning,
directly echoed in the paper's own framing: **"accurate trajectory predictions do not guarantee correct
identification of regional dependence structures"** — i.e. good forecasts don't validate the recovered
"who-infects-whom" structure, a caution structurally identical in spirit to your target paper's claim that
learned graph attention can look fine on the loss while being uninformative.

#### C. Method
Neural network parameterises latent SIRD compartment logits per region, softmax'd to stay non-negative and
sum to population; a row-stochastic source-composition matrix C (softmax over masked logits B) mixes
force-of-infection across source regions; C is regularised toward a prior C₀ (distance decay / adjacency /
commuting / data-inferred lead-lag) with fixed coupling strength s_C=0.10 and regularisation weight λ_C. Two
hidden layers of width 32, tanh activations (synthetic setup).

#### D. Data & protocol
Synthetic: 4 regions, 90 daily points, coordinates given, known true coupling matrix and α*=0.42, single seed
per reported grid configuration ("multistart confidence intervals are unavailable" — self-reported limitation).
Real-world: 64 Louisiana counties, weekly COVID-19, March 5–Dec 31 2020 (~44 weekly observations), four
forecast origins (Jul 23, Sep 3, Oct 15, Nov 26 2020), six-week test windows per origin, with an **external
infection-pressure input observed throughout the test window** (an oracle setting the authors explicitly flag
as unrealistic for deployment — "an operational system would have to forecast that input separately"). Three
neural random seeds per origin; masked-origin training removes post-origin observations from the loss (a
form of leakage-avoidance by construction, though the oracle external input is a genuine limitation on the
realism of the setup, not a leakage bug per se).

#### E. Baselines
Autoregressive Negative-Binomial (fixed-origin), Local-Only PINN (no cross-county coupling, no external
input), No-County-Network PINN (identity coupling matrix, all other inputs retained). These appear to be
author-implemented reference models rather than re-runs of other groups' published code — i.e. there is no
comparison against any *other paper's* published spatiotemporal model (Cola-GNN, EpiGNN, MSAGAT-Net, etc.),
only internal ablation-style baselines.

#### F. Headline numbers
Synthetic (Table 3): coupling error ~0.094–0.099 with weak/medium distance priors vs. 0.159 with no prior vs.
0.577 with a strongly misspecified prior. Louisiana 64-county (Table 5): Forecast-trained Geo-PINN MSE=11,468,
MAE=57.73, NLL=5.346 vs. Autoregressive-NB baseline MSE=32,957, MAE=70.60, NLL=5.158 (65.2% MSE reduction,
18.2% MAE reduction, but the NB baseline actually has *better* NLL — Geo-PINN loses on calibrated
log-likelihood despite winning on point-error metrics, a genuinely interesting and honestly reported tension).
15-county controlled-geography test (Table 6): Neighbor-Network MSE=29,808 vs. No-County-Network MSE=31,999
(only 6.85% MSE gain, 3.1% MAE gain) — i.e. the actual "does the geographic coupling help" effect size on real
data is modest.

#### G. Ablations
Prior type (distance weak/medium, identity, anti-distance, misspecified diagonal); observation channel (clean,
noisy, sparse, reporting-biased, cases-only — Table 4, showing cases-only/reporting-biased observation
substantially degrades the recovered transmission scale α̂ and blows up data loss, a leakage/robustness-style
stress test); model components (Local-Only, Closed-System, Frozen-Input, Balanced variants); geography (a
12-prior sensitivity sweep including commuting, lead-lag, placebo, and unregularised variants — Table 6B).

#### H. How written
Careful, self-limiting paper: explicitly distinguishes three tiers of claim strength — "fitted composition"
(the matrix fits the data), "predictive geography" (ablating the matrix hurts forecasts), and "recovered
coupling" (the matrix reflects real contact/mobility structure) — and states plainly that the Louisiana
results support only the second, weakest tier, not the third. This three-tier claim-strength framework is a
genuinely useful rhetorical/methodological device that your target paper could adopt or cite directly, since
it maps almost one-to-one onto the "attention collapses to uniform pooling" argument: a graph module can be
*predictively* useful (ablating it hurts RMSE) without the learned weights being a *correct* recovered
structure — GeoID-PINN's own Louisiana result (Table 6: only ~7% MSE gain from real geography vs. identity
coupling) is itself a fairly weak "predictive geography" signal, arguably closer to "doesn't clearly help
much" than "clearly helps."

#### I. Red flags
- Single seed per synthetic configuration — "multistart confidence intervals are unavailable," an explicit,
  candid admission of a real statistical-rigor gap.
- Oracle external-input assumption in the Louisiana test windows (input observed throughout the test period,
  not forecast) — makes the real-data results not directly comparable to a genuine forecasting deployment.
- No comparison to any external spatiotemporal GNN baseline (Cola-GNN/EpiGNN/MSAGAT-Net-style) — only internal
  ablations, so its "geography helps" claim isn't benchmarked against the literature's existing graph models.
- Worse NLL than the simpler NB baseline on real data despite better point-error metrics — a genuine, honestly
  reported inconsistency between calibration and accuracy that the paper doesn't fully resolve.

#### Limitations (as extracted, verbatim fragments)
"Each synthetic grid uses one seed, so multistart confidence intervals are unavailable"; "Edge-wise profile
losses were not completed"; Louisiana results are retrospective, "external input observed throughout each
test window," "an operational system would have to forecast that input" separately; "no edge-level truth" for
validation in Louisiana; the model does not "recursively roll the SIRD equations forward"; aggregate
county-week counts prevent "person-level inference"; the matrix cannot be claimed to identify "individual
contacts, literal mobility, or causal transmission pathways."

#### J. Relevance to target paper (i)–(v)
(i) lead-h protocol: not directly about multi-horizon pooling (single 6-week test window per origin, evaluated
presumably across the window jointly) — limited direct overlap, though the four-origin retrospective design is
itself a fairly small/narrow evaluation basis worth contrasting with your multi-seed, pooled-DM-test framing.
(ii) graph attention collapse — **this is the single most relevant paper in the group for this claim.** Its
own real-data ablation (Table 6: real geography only ~7% better than identity coupling) is close to your own
finding that spatial structure is barely contributing, and its explicit three-tier claim-strength framework
("fitted / predictive / recovered") is a ready-made vocabulary for stating precisely what MSAGAT-Net's
attention entropy≈1.0 finding does and doesn't establish. Strongly recommend citing this paper for that
framework, regardless of whether its own SIRD/mobility setting is used elsewhere.
(iii) graph-free common-factor model: GeoID-PINN's identity-coupling ablation (effectively a graph-free
variant) is its own strongest baseline by the numbers (6.85% MSE gap only) — further evidence that graph-free
alternatives are competitive in this broader literature, not just within MSAGAT-Net.
(iv) power analysis/DM/conformal: none present; single-seed synthetic experiments and 3-seed real-data
experiments with no formal significance testing are exactly the kind of under-powered evaluation your target
paper's power-analysis argument is aimed at.
(v) daily COVID panels test>validation: Louisiana is weekly, not daily, and a single 2020 wave — no direct
overlap with NHS/LTLA/Australia, though the "oracle external input" caveat is a useful contrast to your
own leakage-audit rigor (AGENTS.md's "leakage audit verdict is clean" standard is a stronger bar than
GeoID-PINN sets for itself here).

---

### 6. Dudley, Magdaleno, Harding, Sharma, Martin, Eisenberg — "Mantis," arXiv 2508.12260 (v5, 13 Apr 2026)

#### A. Bibliographic
v1 17 Aug 2025, v5 13 Apr 2026 (five revisions — unusually iterated for this group, suggesting substantial
post-submission rework). cs.AI/q-bio.QM, CC BY-NC-SA 4.0 (non-commercial license — a real constraint on reuse
of code/weights if that matters for comparison purposes). 11 pages, 4 figures. No code link surfaced in the
extraction; given the training procedure (400M+ simulated days, presumably large simulator + model), whether
weights/code are publicly released was not confirmed.

#### B. Claims (as extracted)
Mantis is "trained exclusively on mechanistic simulations rather than actual disease data," claimed to enable
prediction "across diseases, regions, and outcomes, even in settings with limited historical data," reportedly
outperforming "all 78 comparative models when assessed against early COVID-19 pandemic data" and generalising
to diseases absent from its training simulations, which the authors interpret as evidence it "learned
fundamental contagion principles rather than disease-specific patterns."

#### C. Method
Sequence-to-sequence CNN-Transformer hybrid: multi-scale convolutional embedding of up to 112 weeks of
history, plus static/temporal covariate embeddings (disease type, population, temporal signals); hybrid
CNN-Transformer encoder (local conv patterns + global attention); an "epidemic pattern memory bank" for
retrieval-style pattern matching (conceptually adjacent to, but architecturally distinct from, TERN's
fast-weight memory — Mantis's memory is a retrieval bank over simulated patterns, not a per-step decaying
associative memory); autoregressive quantile decoder with cross-attention and a GRU update, teacher-forced
during training, weighted quantile loss, AdamW. Trained on >400M simulated days across three mechanistic
families: human-to-human (stochastic SEIR-variants, thousands of parameter combinations), vector-borne
(host-vector, Aedes-inspired), environmental (dual-route contact+water, cholera-inspired), with simulated
observation artefacts (underreporting, delays, day-of-week effects, multiplicative noise). **No real-world
data used during pretraining at all** — this is the paper's central architectural bet, and the "262% error
increase when trained on real COVID data with rolling retraining instead of simulation" ablation (see G) is
offered as direct support for it.

#### D. Data & protocol
Evaluation-only real-world data: 16 diseases, 39+ forecasting tasks, ~150,000 individual forecasts. Headline
tasks: COVID-19 mortality (US states, early pandemic, benchmarked against CDC Forecast Hub submissions
directly), influenza hospitalisation (local scale). Distribution-shift stress tests: dengue (Brazil), scarlet
fever (historical), hepatitis B (chronic — a genuinely out-of-distribution test since training simulations are
"acute outbreak" oriented), smallpox (eradicated disease, presumably tests generalisation to a disease absent
from any modern surveillance training distribution), ILI syndromic, plus Thailand/DRC Ebola/Africa
Mpox/Ethiopia cholera. Protocol: zero-shot, no fine-tuning on any evaluation dataset; "strict temporal
validation using data as available at forecast time," rolling-retraining baselines for fair comparison (i.e.
the *baselines* get to retrain as new data arrives, but Mantis itself does not retrain — an asymmetry the
paper treats as a selling point of the simulation-pretraining approach, but which also means Mantis is not
adapting online the way TERN's online-refit/online-blend mechanisms do).

#### E. Baselines
78 total: CDC Forecast Hub COVID-19 submissions (real operational ensemble models, presumably copied from
the Hub's published results, not re-run) plus reimplemented naive-persistence, ETS, SARIMA, LSTM (rolling
retrain), Chronos (zero-shot foundation model, a direct point of overlap with TERN's Chronos-2 comparison —
worth cross-referencing whether Mantis and TERN report consistent relative standings for Chronos-family
models, though they evaluate different diseases/geographies so a direct comparison isn't possible without
further digging).

#### F. Headline numbers
Relative MAE/WIS (normalised to naive baseline, lower better) and 50%/95% coverage, across diseases: COVID-19
(state) Mantis rel.MAE 0.65 vs. CDC ensemble 0.66 (Mantis slightly better on MAE, CDC ensemble slightly better
on WIS 0.61 vs 0.63 — a genuine near-tie, not a blowout, and honestly reported as such); Flu hosp. (local)
Mantis 0.85/0.86; Dengue 0.75/0.80; Smallpox 0.83/0.84; ILI 0.83/0.88; Scarlet Fever 0.80/0.81; Hepatitis B
0.85/0.84. Coverage rates (50%/95%) generally close to nominal (e.g. 0.51/0.91 for COVID), suggesting
reasonably calibrated intervals across very different disease regimes — a meaningfully strong calibration
claim if it holds up.

#### G. Ablations
Architecture-capacity ablation: swapping the hybrid CNN-Transformer for a 200×-smaller LSTM with identical
training data increases MAE by 89% — evidence the architecture, not just the simulation data, matters.
Pretraining-source ablation: training the *same* architecture exclusively on real COVID data with rolling
retraining (i.e. removing the simulation-pretraining, keeping everything else) increases error by 262% versus
the simulation-pretrained model — the paper's central ablation, and a strong, if single-number, piece of
evidence for the "simulation pretraining beats real-data-only training" thesis.

#### H. How written
Foundation-model framing (large-scale simulation pretraining + zero-shot generalisation across diseases),
benchmarked against an unusually large baseline set (78 models) including a real operational forecasting
hub (CDC), which is a strong, credible external comparison rarely available to academic epidemic-forecasting
papers. Five revisions (v1→v5) over ~8 months suggests significant post-hoc strengthening, possibly in
response to review. Explicit distribution-shift stress-testing (eradicated diseases, chronic diseases, novel
geographies) is a genuinely distinctive and valuable evaluation design choice relative to the rest of this
group.

#### I. Red flags
- CC BY-NC-SA license and no confirmed public code/weights in this session — reduces ability of outside
  groups (including yours) to independently verify or reuse.
- The "262% worse when trained on real data only" ablation is a single comparison on one disease/setting per
  the extraction (unclear if replicated across multiple diseases) — a dramatic number that would benefit from
  more replication before being treated as a settled finding.
- COVID-19 near-tie with the CDC ensemble (0.65 vs 0.66 MAE, CDC actually better on WIS) is reported honestly
  but is a much more modest win than the "outperformed all 78 comparative models" framing in the abstract
  implies at first read — worth checking the exact wording/ranking methodology (best on how many of 78, on
  which metric, on which subset of tasks) before citing the "beat all 78" claim uncritically.
- Explicit self-reported limitation that spatial coupling is largely absent (single-location forecasts;
  "preliminary evaluation showed ~5% MAE improvement with adjacent-state covariates" — i.e. even the authors'
  own preliminary check suggests spatial structure has a real but modest effect size, congruent with the
  GeoID-PINN and MSAGAT-Net findings above).

#### Limitations (as extracted)
Performance depends on simulation diversity/accuracy; systematic simulator biases propagate to the trained
model; single-location forecasts lack explicit spatial coupling (mobility/inter-region dynamics not modeled,
though "preliminary evaluation showed ~5% MAE improvement with adjacent-state covariates"); trained on acute
outbreak simulations, so generalisation to chronic persistence dynamics (hepatitis B) is "limited despite
competitive performance"; unclear whether some epidemic regimes require explicit simulation representation
versus emerging from sufficiently diverse simulation coverage; evaluation used historical surveillance data
with its original reporting gaps/quality issues, affecting all models equally but still a data-quality caveat.

#### J. Relevance to target paper (i)–(v)
(i) lead-h protocol: horizons are 2/4/6/8 weeks; whether each is scored using a lead-h-only forecast or a
single multi-horizon output evaluated per lead was not confirmed by the extraction — worth a follow-up check,
since Mantis's autoregressive decoder plausibly emits all horizons from one forward pass (unlike TERN's
per-horizon retraining), which would make it structurally similar to the "one model scored at multiple
horizons" pattern your paper is scrutinising — but I could not confirm this from the extraction alone.
(ii) graph attention collapse: Mantis explicitly lacks spatial coupling (self-reported limitation, ~5% MAE
gain from simple adjacent-state covariates in a preliminary check) — another independent data point, on a
completely different architecture/training paradigm, that spatial machinery contributes only a small,
single-digit-percent effect size in this literature when it's tested honestly.
(iii) scale-equivariant, graph-free common-factor model with seasonal memory: Mantis is itself graph-free and
relies on simulation diversity plus a retrieval-style "epidemic pattern memory bank" rather than an explicit
seasonal-reference term — a different route to the same general idea (give the model access to "shapes seen
before" beyond the immediate window) as TERN's climatology/season-embedding and your own seasonal-memory
proposal; worth citing as a third distinct implementation of "pattern memory beyond the input window" in
recent literature (TERN's explicit seasonal reference, Mantis's simulation-derived memory bank, and your
target common-factor/seasonal-memory model).
(iv) power analysis/DM/conformal: Mantis does use Diebold-Mariano significance testing (explicitly named in
its evaluation protocol per the extraction) — this is the one paper in the whole group besides possibly TMLR's
that explicitly reports a formal significance test; worth citing as a positive precedent for DM-testing in
this exact literature.
(v) daily COVID panels, test-levels-exceed-validation: no NHS/LTLA/Australia-scale daily panel overlap; the
COVID-19 evaluation is US-state-level, weekly-aggregated mortality against the CDC Hub, not daily case
panels — limited direct overlap, though the "zero-shot, strict temporal validation using data as available at
forecast time" framing is broadly consistent with your no-look-ahead requirement.

---

### Synthesis: what TERN does and does not cover relative to (i)–(v) (~400 words)

TERN is the strongest structural match to your target paper among the six, and the closest thing to a direct
competitor, but it covers a narrower slice of your five contributions than a first read of the abstract
suggests. On **(i) lead-h evaluation correction**, TERN already does what you're arguing others should do: it
retrains one model per horizon and scores each lead-h-only, with pooled-within-horizon RMSE averaged (not
pooled) across the four horizons — confirmed directly from `scripts/export_tables.py`. This means TERN is not
a counter-example to fix; it's independent confirmation from a concurrent paper that lead-h-only scoring is
the defensible convention, strengthening your protocol-correction argument by precedent rather than needing
TERN itself as a contrast case. TERN offers **no equivalent to (ii)** — spatial coupling is one causal
attention layer over regions (`region_attention`, optional `adjacency_bias`), ablated in Table 2 with modest
effect sizes (no adjacency bias only costs ~1 RMSE point on US-States), but TERN never inspects whether that
attention is doing anything selective (no entropy analysis, no attention-map inspection); this is a genuine
gap your paper fills that TERN does not, and the region-attention ablation's small effect size is itself
weak circumstantial support for your "graph/attention machinery contributes little" thesis. On **(iii)**,
TERN and your proposal diverge in a complementary way: TERN keeps per-region delta-rule memories plus an
explicit seasonal-reference/climatology correction with horizon-dependent shrinkage (more trust in the
learned correction at short leads, more trust in climatology at long leads) — structurally similar in spirit
to a horizon-monotone persistence/structure gate, but implemented as a scalar shrinkage schedule rather than a
learned, scale-equivariant common factor; TERN is not graph-free (region attention remains), and its
"seasonal reference" is influenza-specific two-season climatology, not a generalised common-factor decomposition.
This is the clearest point of both overlap and contrast to foreground in your paper: TERN independently
arrived at "shrink toward a persistence-like reference more as the horizon grows," which corroborates your
horizon-monotone-gate intuition, while differing in mechanism and in retaining graph machinery you argue is
inert. On **(iv)**, TERN is a clear gap: five seeds for headline numbers but only three for ablations, no
variance reporting beyond seed-median-vs-mean substitutions for one unstable baseline, and zero significance
testing anywhere in the paper or its table-export code — exactly the evaluation weakness your power-analysis/
pooled-DM-test/conformal-interval framing is positioned against. On **(v)**, TERN never touches COVID or daily
data at all — three weekly influenza panels capped at 49 regions, nothing resembling NHS-7/LTLA-372/Australia-8
daily panels or your test-exceeds-validation framing. Net: TERN is a rigorous, well-executed concurrent
competitor on protocol mechanics (i) and gives circumstantial support to (iii), but is silent on (ii), weak on
(iv), and entirely absent on (v) — your paper's daily-COVID, graph-attention-diagnostic, and statistical-rigor
contributions remain clearly differentiated from it.

---

## Group 5 — user's local PDFs in `doc/GNN forecasting/`

Read cover to cover (all pages, via the PDF `pages` parameter) on 2026-09-29.
An earlier pass exists at `doc/audit-2026-08-24/04-user-papers-folder.md`
(items 1–9 there cover 8 papers + Fritz et al., read for a different purpose
— "could this run on our benchmark"). This note independently re-reads the
7 papers assigned to this reader, verifies every number quoted below against
the PDF, and organizes them under the A–J rubric requested for this pass.
Where the audit's numbers were checked they matched; corrections/additions
are flagged explicitly. STAN (files/1624) was skipped per instructions
(covered by another reader).

**File → paper mapping, confirmed by reading, with one naming correction:**
the file `files/2016/Jun Zhao et al_2023_Advances in spatiotemporal graph
neural network prediction research.pdf` is mislabeled in its filename — the
actual paper (confirmed from the title page) is **single-author Yi Wang
(2023)**, *International Journal of Digital Earth* 16(1):2034–2066. "Jun
Zhao et al." does not appear as an author anywhere in the PDF; this looks
like a citation-manager metadata error, not a different paper. Likewise
`1-s2.0-S095741742502010X-main.pdf` is Yin, Qiu, Fang, Wang, Dong, Ge (2025),
*Expert Systems With Applications* 291:128391 — a business-process-mining
paper, not epidemic forecasting (flagged below).

---

### 1. Yin, Qiu, Fang, Wang, Dong & Ge (2025), STGNN

#### A. Bibliographic
*Expert Systems With Applications* 291 (2025) 128391.
DOI 10.1016/j.eswa.2025.128391. Authors: Jun Yin, Aoxue Qiu, Lin Fang,
Nianxin Wang, Chen Dong, Shilun Ge (Jiangsu University of Science and
Technology; Hangzhou Dianzi University). **No code link** — "Data
availability: Data will be made available on request." Proprietary dataset
from one Chinese shipbuilding company.

#### B. Claimed contributions (verbatim, from the intro's bullet list)
> "Complicated linked process network models perspective: We extend the
> focus beyond individual processes to predict business process performance
> from a network models perspective... Spatial-temporal graph neural network
> (STGNN): We introduce a Spatial-Temporal Graph Neural Network (STGNN)
> model that captures both temporal and spatial characteristics... Empirical
> validation: We evaluate our proposed method on a dataset of business
> processes with typical parallel events in a real manufacturing company."

#### C. Method (5–8 lines)
Not epidemic forecasting — it predicts the future execution frequency
(count, aggregated per 24h) of 14 shipbuilding sub-processes. Pipeline: (1)
mine a directed **process network** from event logs (process-mining
transition extraction, 5 major activities refined into 14 nodes, edges =
observed transitions, filtered at a 20% trace-frequency threshold); (2)
**temporal attention** — an `X'^{t-1'} = E' X'^{t-1}` self-attention over the
T time steps (Eqs. 1–3); (3) **spatial attention** — an analogous attention
over the N=14 nodes to catch *indirect* (non-adjacent) dependencies (Eqs.
4–6); (4) a **GAT layer** (multi-head, LeakyReLU, softmax) to aggregate
*direct* neighbor influence (Eqs. 7–8); (5) MLP output head. Loss: MAE.

#### D. Data & protocol
Single company, single 14-node process network, 2014–2020 event log
(5,552,615 total events across 5 documents). Node feature = execution
frequency aggregated in **24-hour windows**. Sliding-window sample
construction (input window / target window, Fig. 9) — window/horizon
lengths not stated as a fixed number in the text shown, but Table 3 gives
timesteps=7 for input, and separate rows for prediction horizons 1/3/5/7
(days) — i.e., horizons 1, 3, 5, 7 days, each with its own train/test split
(nodes=14 fixed, "Feature"=1). **Split: 80:20, single split, not stated as
chronological or random** ("The dataset is divided into training and
testing sets in an 80:20 ratio, with 80% allocated for model training and
the remaining 20% reserved for evaluating the model's performance" — no
explicit statement that the split is chronological, though a sliding-window
construction over an event log strongly implies it is). **No seeds
reported, no std/variance, no significance test anywhere in the paper.**
Metric: MAE only. **Naive baseline: HA (historical average) only** — no
persistence/last-value baseline.

#### E. Baselines; re-run vs copied
HA, LSTM, GRU, CNN — all four are **run by the authors on their own
proprietary dataset** (there is nothing to copy from; no public benchmark
numbers exist for this task). Four baselines total, three of which are deep
sequence models with no explicit spatial/graph component; only HA is
non-deep.

#### F. Headline numbers
Table 8, MAE by horizon (1/3/5/7 days): HA 321.592 (constant); LSTM
224.736/229.863/238.243/237.602; GRU 222.237/227.385/239.715/239.481; CNN
207.018/218.770/226.005/229.417; STGNN-NoGraph
212.091/226.112/237.861/242.003; **STGNN 202.210/215.493/217.908/224.561**.
Verified: STGNN beats HA/LSTM/GRU/CNN by 37.12%/10.03%/9.01%/2.32% at h=1
(all four percentages recompute exactly from the table). No datasets
overlapping ours (no Japan-Prefectures/US-Regions/US-States/Australia/UK).

#### G. Ablations/interpretability
One ablation: STGNN vs **STGNN-NoGraph** (GAT branch removed). MAE
degrades by **4.66% / 4.70% / 8.39% / 7.21%** at h=1/3/5/7 (recomputed and
confirmed exactly from Table 8) — the explicit-adjacency (GAT) branch's
value *grows with horizon*, matching this project's own horizon-threshold
finding. Interpretability claims are narrative only: the paper reads off
attention/correlation heatmaps (Fig. 8, Pearson correlation across the 14
processes) and a qualitative "drastic change" case study (Table 9: STGNN
tracks large jumps, e.g., predicted 1112 vs actual 1046 on 2014‑11‑29, where
GRU/LSTM/CNN are 124/109/247) — no faithfulness test, no ground-truth
attention validation.

#### H. Writing / venue-style notes (imitable for a CBM/Elsevier submission)
Structure: Intro (problem framing with an industry vignette) → Related Work
(2 clean subsections: process-performance prediction; GNNs+applications) →
Preliminaries (4 numbered formal Definitions: Event/Trace, Event Log,
Business Process Network, Process Performance Prediction) → Methodology
(motivating figure first, Fig.1, "spatial-temporal dependence" intuition
before any equation) → Experiments (dataset description, heavy use of
domain figures — Fig. 4 activity-relationship diagram, Fig. 5 multi-route
diagram, Fig. 6 final network, Fig. 7 month-by-month network snapshots, Fig.
8 correlation heatmap — *before* any model numbers) → Results (quantitative
table, then a "drastic-change stage" qualitative case study) → Conclusion &
future work → CRediT statement → Data availability → Funding →
Declaration of competing interest → References. Notable ESWA-house-style
elements worth imitating: (1) each of the paper's three contributions is a
one-paragraph bullet, restated near-verbatim in the abstract; (2) every
equation is immediately followed by a one-sentence plain-English gloss of
each symbol; (3) limitations are split explicitly into two labeled
subcategories in the Conclusion ("model characteristics" vs "data
characteristics") rather than a vague single paragraph; (4) a formal
CRediT authorship table and funding/competing-interest/data-availability
boilerplate immediately after Conclusion, before References — standard
Elsevier requirement, easy to miss if drafting from a NeurIPS/ICML template.

#### I. Red flags
No code, proprietary single-company data (irreproducible by construction).
Single train/test split, **no seeds, no variance, no significance testing,
no confidence intervals anywhere**. Only one naive baseline (HA) — no
persistence baseline, which is the single most damning baseline-hygiene gap
in the set (this project's own AGENTS.md and prior audit both flag
persistence baselines as mandatory). N=14 nodes, single graph, single
domain — no external validity claim attempted or warranted. Graph is
*learned from event logs*, not epidemiological — spatial semantics differ
totally from geographic adjacency.

#### J. Relevance to the new paper
Not a methods precedent for epidemic forecasting or graph-attention
collapse — it's a **venue-fit calibration sample** for Computers in Biology
and Medicine (a similarly mid-tier Elsevier "Expert Systems"-family journal
with the same reviewer culture): it shows what such venues accept (a
well-motivated domain reframing + assembled off-the-shelf GNN blocks + one
proprietary dataset + one ablation + no seeds) — a bar this project's paper
would clear by a wide, demonstrable margin. Also directly useful as a
*writing template*: the explicit Definitions section, the plain-English
equation glosses, and the split "model vs data" limitations subsections are
concrete, copyable conventions for a CBM submission.

---

### 2. Kosma, Nikolentzos, Panagopoulos, Steyaert & Vazirgiannis (2023), GN-ODE

#### A. Bibliographic
*Transactions on Machine Learning Research*, 08/2023. École Polytechnique /
IP Paris. OpenReview `yrkJGneOuN`. **No code repository link found anywhere
in the paper** (not in the main text, appendix, or acknowledgements — this
is itself notable for a TMLR paper).

#### B. Claimed contributions (verbatim)
> "In this paper, we propose a novel deep neural network architecture for
> modeling and predicting spreading processes... we are the first to apply
> task-specific neural ODEs for the SIR model, intending to advance the
> learning capabilities of a standard neural network model."
> "...paving the way for the extensive application of interpretable neural
> networks in the field of epidemic spreading."

#### C. Method
Individual-based SIR on a **fixed contact-network adjacency A** (β, γ
scalar, uniform, sampled per-instance not learned). Node-level 3-state
(S/I/R) probability vectors, initialized via a linear+ReLU embedding of the
binary initial condition, then iteratively refined by a black-box ODE
solver (Euler, step 0.5) implementing `dS/dt=-β(AI)⊙S, dI/dt=β(AI)⊙S-γI,
dR/dt=γI` with the states additionally passed through learned
sigmoid-gated linear layers at each solver step. Output: softmax
3-vector per node/timestep via adjoint-based backprop (constant memory).
Ground truth = 10⁴ Monte Carlo SIR simulations, 20 steps, on 8 real social
networks (34–75,877 nodes).

#### D. Data & protocol
8 networks (karate, dolphins, fb-food, fb-social, openflights, Wiki-Vote,
Enron, Epinions), Table 1 gives exact node/edge/transitivity/density/max
degree for each. 200 instances per dataset (β,γ ~ U[0.1,0.5], 2 random seed
nodes infected), **60:20:20** split. **5 repeats, mean ± std reported**
(Table 2). Metric: MAE across all nodes/states/20 timesteps (exact formula
given, Eq. after "Evaluation metric"). Hyperparameters via grid search on
validation loss (lr, batch size, hidden dim). Naive baseline: **DMP**
(Dynamic Message Passing — an exact/asymptotically-exact combinatorial
method, not learned) plays this role; there is no persistence baseline in
the epidemic-forecasting sense since the task is marginal-probability
estimation, not time-series extrapolation.

#### E. Baselines; re-run vs copied
All baselines (DMP, GCN, GIN) run by the authors themselves on their own
generated datasets — nothing copied from other papers. GN-ODE best on 5/8
datasets; DMP wins the three largest (Wiki-Vote, Enron, Epinions), at the
cost of ~10x inference time (Fig. 2b). Ablation vs a fixed **ODE-RK**
(Runge–Kutta solve of the *untrained* SIR ODE system, Table 2, Appendix):
MAE 0.09608 (karate) / 0.10653 (dolphins) / 0.19109 (fb-food) / 0.11061
(fb-social) / 0.16087 (openflights) / 0.12287 (Wiki-Vote) / 0.16572 (Enron)
/ 0.15917 (Epinions) vs GN-ODE 0.05631±0.00062 / 0.01527±0.00049 /
0.01924±0.00111 / 0.01089±0.00102 / 0.02000±0.00145 / 0.04173±0.00287 /
0.04885±0.00125 / 0.05915±0.00224 — **confirms the audit's read exactly:
the neural refinement, not the SIR prior, does essentially all the work**
(3–17x error reduction over the untrained mechanistic system).

#### F. Headline numbers
Not comparable to our forecasting benchmarks (no Japan/US-Regions/etc.) —
this is node-level marginal-probability estimation on social/contact
graphs, not epidemic-count time series forecasting.

#### G. Ablations/interpretability
Two genuinely well-designed generalization protocols, both worth stealing:
(i) **OOD generalization** — bin the 200 (β,γ) instances into 5 bins by
value, train on bins 2–4 only, test on bins 1 and 5 (Fig. 3–5, 8):
GN-ODE dominates GCN/GIN at every OOD point, though its own OOD error rises
on the three largest graphs. (ii) **Cross-graph transfer** — train on
5 small networks jointly (karate…openflights), test zero-shot on 3 much
larger unseen networks (Wiki-Vote 7066 nodes, Enron 33696, Epinions 75877):
GN-ODE's single-graph-trained vs multi-graph-trained gap is small (Fig. 6b)
where GIN's is dramatic — genuine size-transfer evidence, run 5x with error
bars. **Interpretability is claimed in the abstract ("paving the way for...
interpretable neural networks") but validated only by one qualitative
figure** (Fig. 7: a side-by-side Monte-Carlo-vs-GN-ODE infection heatmap on
the 34-node karate network at t=0,4,8,12) — no quantitative faithfulness
test, no recovered-parameter check against the true β/γ used to generate
each instance.

#### H. Writing
TMLR house style: no page limit, appendix carries the full derivation
(exact vs closed vs independence-assumption SIR systems, Appendix A.1–A.2),
an extra ODE-RK ablation, and 2 pages of supplementary OOD scatter plots.
Discussion/Conclusion section is candid about **for whom this is useful**
(replacing slow DMP-style combinatorial inference) rather than oversized
epidemiological claims. No explicit "Limitations" heading — concessions are
folded into the Discussion prose (inference cost rises on the 3 largest
networks; GIN not the right inductive bias for this specific task).

#### I. Red flags
Individual-based SIR requires **known contact-network topology at the
individual level, known/sampled β and γ, and Monte Carlo ground truth** —
none of which exist for weekly counts-only epidemic surveillance data. No
comparison to any epidemic-forecasting GNN (Cola-GNN, EpiGNN, STAN, MPNN)
despite citing Panagopoulos's own MPNN+TL paper. β,γ assumed **uniform
across all edges/nodes** — a strong homogeneity assumption inconsistent with
real heterogeneous transmission.

#### J. Relevance
Confirms the audit's framing: **do not build another SIR-simulation
surrogate** — this slot (node-level probability estimation on individual
contact networks, SIR baked in as the exact forward equation) is occupied
and is structurally the wrong task for counts-only surveillance data. The
two transferable methodological ideas are the **OOD-bin generalization
protocol** and the **cross-graph-size transfer protocol** — both directly
adaptable as robustness checks for a scale-equivariant, graph-free
common-factor model (test on regions/population scales never seen in
training, not just held-out time). The paper is also a clean illustration
of the project's central critique: an "interpretable" claim in the abstract,
backed by one visual example and never quantified — exactly the failure
mode this project's Paper B calls out for attention mechanisms.

---

### 3. Shi, Zhang & Morris (2022), PAN-cODE

#### A. Bibliographic
*Journal of the American Medical Informatics Association* 29(12):2089–2095.
doi:10.1093/jamia/ocac160. Brief Communication. University of Toronto /
Vector Institute / Memorial Sloan Kettering. **Code:
https://github.com/morrislab/PAN-cODE** (confirmed present in text, under
"Discussion").

#### B. Claimed contributions (verbatim)
> "We present Pandemic conditional Ordinary Differential Equation
> (PAN-cODE), a deep learning method to forecast daily increases in pandemic
> infections and deaths. By using a deep conditional latent variable model,
> PAN-cODE can generate alternative caseload trajectories based on alternate
> adoptions of NPIs... We demonstrate that, despite using less detailed data
> and having fully automated training, PAN-cODE's performance is comparable
> to state-of-the-art methods on 4-week-ahead and 6-week-ahead forecasting."

#### C. Method
Conditional Latent ODE. A GRU-ODE encoder reads the daily infection/death
counts (+ covariates) up to the forecast date and outputs `(μ_z0, σ²_z0)`
for a variational posterior over an initial latent state `z0`. `z0` is
**concatenated with 4 OxCGRT NPI-stringency indices** (Stringency Index,
Government Response Index, Containment & Health Index, Economic Support
Index, all evaluated *at the forecast date*) to form the augmented state
`z̃0`. A Neural ODE (`f_φ`) evolves `z̃0` forward; an autoregressive decoder
(a simple linear layer combining the previous timestep's prediction with
the current latent state) maps the trajectory back to daily
infection/death counts, restricting the maximum step-to-step change.
Trained end-to-end by maximizing the ELBO (Adam).

#### D. Data & protocol
US state + county daily COVID-19 case/death counts from GCP Open-Data
(~2,500 trajectories, since Feb 2020); **7-day rolling average** applied to
de-noise; `log(x+1)` transform for stability. 4 **conditioning features**
= OxCGRT indices (national-level per country; per-state/county for the US
subset), plus daily temperature as a covariate; **14-day lag** applied
between NPI features and deaths to account for delayed effect. **Random
epoch-forecast-date sampling** during training (exposes the model to many
(date, 4-week-ahead-target) pairs); validation cutoff withholds all data
after a fixed date. **Two evaluation forecast dates: Dec 28 2020 and Mar 8
2021; horizons 4 and 6 weeks.** Metric: **median absolute error (MAE) and
mean rank**, computed per-method across the 51 ranked lists (50 states +
DC). Significance: **Wilcoxon signed-rank test, p<.05, vs the best
per-metric method** — methods not significantly worse than the winner are
marked in bold italics in Table 1 (a genuinely useful convention).
**No seed count is stated anywhere in the paper** — this is a real gap.
Naive baseline: **Baseline(PrevWeek)** (persistence-style, repeating the
previous week's trajectory) — present and reported, unlike several other
papers in this set.

#### E. Baselines; re-run vs copied
~20 methods from the **COVID-19 Forecast Hub** (JHU_IDD-CovidSP,
Covid19Sim-Simulator, IowaStateLW-STEM, Columbia_UNC-SurvCon, UCLA-SuEIR,
USC-SI_kJα, JHUAPL_Bucky, GRU-ODE, VAE-GRU, Google_Harvard-CPF,
UMass-MechBayes, CU-select, COVIDhub-ensemble, Caltech-CS156,
UCSD_NEU-DeepGLEAM, UA-EpiCovDA, TTU-squider, Mean COVID Hub) — **numbers
copied from the public Forecast Hub submissions**, not re-run by the
authors; only PAN-cODE, GRU-ODE, and VAE-GRU appear to be run in-house
(GRU-ODE is PAN-cODE's own unconditioned ablation).

#### F. Headline numbers (Table 1, verified against the PDF)
Dec 28 2020, 4wk: **PAN-cODE MAE 167** (mean rank 7.39) vs best,
Google_Harvard-CPF 118 (bold-italic-tier, i.e. not sig. worse than PAN-cODE
per the Wilcoxon test — PAN-cODE is competitive, not dominant, at 4wk).
Dec 28 2020, 6wk: **PAN-cODE 207** (rank 3.63, underlined = best) vs
JHU_IDD-CovidSP 312, Covid19Sim-Simulator 321, Baseline(PrevWeek) 343.
Mar 8 2021, 4wk: PAN-cODE 93 vs Covid19Sim-Simulator 67 (best),
Google_Harvard-CPF 86. Mar 8 2021, 6wk: **PAN-cODE 80** (rank 4.41,
underlined = best) vs JHUAPL_Bucky 109, Google_Harvard-CPF 86,
Baseline(PrevWeek) 295. Text states: "we find that the error in PAN-cODE's
forecasts is not significantly larger than that of the best model in every
metric. Furthermore... PAN-cODE provides significantly lower error than all
other methods" at the March 2021 6-week task. Unseen-country generalization
(Table 2, % error, from Dec 28 2020): PAN-cODE 4wk: Canada −10.4%, UK 8.2%,
India 7.2%, Russia 0.4%, vs Baseline(PrevWeek) 39.0/66.7/63.1/53.4% and
GRU-ODE 474.4/36.4/144.8/−5.5%. At 6wk PAN-cODE degrades to
−51.4/−21.6/−55.3/−55.2% (large under-forecast) — the paper still calls
these "reasonable projections," which overstates what a >50% error implies.
No overlap with this project's benchmark geographies.

#### G. Ablations/interpretability
No formal ablation table beyond the GRU-ODE (no-NPI-conditioning) and
VAE-GRU comparisons already in Table 1/2 — GRU-ODE performs far worse
(e.g. 644 vs 207 at Dec-2020 6wk), which is effectively the "no
conditioning" ablation and is the load-bearing evidence for the paper's
central claim. Interpretability: LIME feature importance (Supplementary
Appendix E) — validated only by "agreement with intuition" and an
explicit caveat that other feature-importance methods (BorutaSHAP) "might
find different relationships." Counterfactual NPI trajectories (Fig. 3:
Mississippi state and Fresno County) validated only by visual plausibility
against a 95% CI band from 100 sampled latents — no ground truth for the
counterfactual claim exists by construction.

#### H. Writing
JAMIA "Brief Communication" format: **extremely compressed** (7 pages,
Introduction/Background/Method/Experiments-and-Results/Discussion/
Conclusion, most technical detail pushed to Supplementary Appendices A–E).
Every method choice is immediately justified by what it buys practically
("PAN-cODE does not require mobility or hospitalization data," "capable of
natively handling datasets... sparsely or irregularly observed" — the
second claim is **asserted, never tested**). Explicit, honest Discussion
paragraph acknowledging the central limitation:
> "PAN-cODE does not explicitly learn a causal model between NPI stringency
> and future caseload. Building a formal causal model is likely difficult
> due to delayed and noisy reporting, and we leave this as future work."
And the single most load-bearing sentence for this project:
> "By using the Neural ODE, PAN-cODE would be able to fit the dynamical
> parameters of this SIR model using backpropagation..." — an unexecuted
> 2022 TODO describing a renewal/compartmental-plus-neural-ODE fusion.

#### I. Red flags
No graph, no spatial coupling at all — "the graph enters" claim in the
audit is accurate: regions are independent trajectories sharing only
network weights. OxCGRT conditioning is **national-level for most
countries**, making the core contribution (alternative-scenario generation
via `I_fc`) untestable on any sub-national dataset without a national NPI
series (this rules it out for LTLA/Australia/Japan-Prefecture-style
datasets as published). No seed count reported. 6-week under-forecasting of
>50% on unseen countries is glossed as "reasonable."

#### J. Relevance
The clearest "closed TODO" in the whole reading list: PAN-cODE's own
Discussion explicitly proposes fusing a Neural ODE with a SIR compartmental
model and never does it — precisely the SIR-embedded-GNN space the earlier
audit already found saturated (STAN, MepoGNN, EARTH, HeatGNN, GN-ODE,
etc.), reinforcing that this axis should **not** be pursued further. Two
things worth borrowing regardless: (i) the **Wilcoxon-signed-rank
"not-significantly-worse-than-best" table convention** (bold-italic tiering
in Table 1) is a clean, reviewer-friendly way to report significance
without collapsing everything to one winner — directly applicable to a
power-analysis/calibrated-intervals paper; (ii) the explicit, understated
framing ("comparable to," "not significantly worse than," never claiming
outright superiority except where the Wilcoxon test actually supports it at
6 weeks) is a good tone model for a paper that is deliberately correcting
inflated claims elsewhere in the literature.

---

### 4. Jin, Zheng, Pan & Chen (2023), MTGODE

#### A. Bibliographic
*IEEE Transactions on Knowledge and Data Engineering* 35(9):9168–9180.
doi:10.1109/TKDE.2022.3221989. Monash University / East China Normal
University / Griffith University. No explicit code-repository URL was
visible in the pages read (footnotes 2–3 on p.9175 link only to two
*dataset* repos, `github.com/laiguokun/multivariate-time-series-data` and
`github.com/liyaguang/DCRNN`, not to a code release for MTGODE itself); the
audit's GitHub link (`github.com/GRAND-Lab/MTGODE`) was not independently
verified from the visible text and should be checked before citing.

#### B. Claimed contributions (verbatim, from Conclusion)
> "By solving the intersecting continuous graph propagation and temporal
> aggregation processes, our method allows the model to learn more
> expressive representations efficiently without relying on graph priors,
> showing better potential in real-world applications... we also
> theoretically analyze the main properties of our method and further
> demonstrate that it is more effective and efficient than the existing
> discrete approaches."

#### C. Method
Two coupled continuous processes, each with a formal Proposition proving a
property of the discrete-to-continuous limit. **Continuous Graph
Propagation (CGP)**: `dH^G(t)/dt = (Â−I)H^G(t)` (Eq. 6), with depth `K` and
integration time `T_cgp` **decoupled** (`K=T_cgp/Δt_cgp, K→∞`) — Property 1
proves this converges (over-smoothing avoided by construction, not by a
regularizer). **Continuous Temporal Aggregation (CTA)**:
`dH^T(t)/dt = P(A(ODESolve(TCN(H(t),t),·,0,...,T_cgp)),0,...,T_cgp),R)` —
gated dilated TCN with kernel widths `{2,3,6,7}` chosen explicitly because
"most... time series data have inherent periods (e.g., 7, 14, 24, 28, and
30)." Graph structure is **learned with no prior** (MTGNN-style
node-embedding product, Eq. 8), sparsified, directed. The two ODEs are
nested: intermediate CTA states feed CGP as evolving initial conditions
(Eq. 15) — "not simply concatenating them end-to-end."

#### D. Data & protocol
5 datasets (Table 2): Electricity (26,304 samples, 321 nodes, 1hr, no
predefined graph), Solar-Energy (52,560, 137, 10min, no graph), Traffic
(17,544, 862, 1hr, no graph), Metr-La (34,272, 207, 5min, **predefined
graph**), Pems-Bay (52,116, 4,732 [*sic*, likely 325 sensors per later
text], 5min, predefined graph). **Single-step**: input length 168, 60/20/20
chronological split, RSE + CORR metrics, "independently repeated ten times
on Linux servers with two AMD EPYC 7742 CPUs and eight NVIDIA A100 GPUs.
Averaged performances are reported" — **10 runs, but no standard deviations
appear in Table 3 or 4** (only Fig. 4/5 sensitivity plots show shaded
std-bands). **Multi-step** (Metr-La/Pems-Bay): input/output length 12,
70/10/20 split, 200 epochs, Euler solver (Metr-La) / Runge-Kutta (Pems-Bay),
MAE/RMSE/MAPE @ horizons 15/30/60 min.

#### E. Baselines; re-run vs copied
Single-step: VARMLP, GRU, LSTNet, TPA-LSTM, MTGNN, HyDCNN, STG-NCDE — run
under a common protocol (adopting LSTNet's [12] published config for
fairness), "we follow [12]." Multi-step: DCRNN, STGCN, Graph WaveNet, GMAN,
MRA-BCGN, MTGNN, STGODE, STG-NCDE.

#### F. Headline numbers (verified against Tables 3–4)
Single-step, Electricity h=3: **MTGODE RSE 0.0736 / CORR 0.9430** vs MTGNN
RSE 0.0745 / **CORR 0.9474 (MTGNN wins on CORR here)** — confirms audit's
claim exactly. Electricity h=12: MTGODE 0.0891/0.9279 vs HyDCNN 0.0921/
0.9285 (MTGNN win on CORR at 12 too, marginal). Multi-step Metr-La 60min:
MTGODE MAE 3.39/RMSE 8.19/MAPE 7.05% (best) vs MTGNN 3.50/8.79 [table shows
3.50 for MAE]/7.30%. Pems-Bay 60min: MTGODE 1.88/4.31/4.31% vs DCRNN
2.07/4.74/4.90%. No datasets overlap this project's benchmarks; input
lengths (168–17,544+ timesteps) are 50–5000x longer than Japan-Prefectures
(348 weekly points), so the statistical regime is entirely different.

#### G. Ablations/interpretability
Table 5 ablation (Electricity/Traffic/Solar-Energy): full MTGODE beats
w/o-GSL (graph structure learning), w/o-CTA, w/o-CGP, and w/o-CGP&Attn on
every cell — e.g. Solar-Energy RSE 0.1686 (full) vs 0.1897 (w/o CGP&Attn).
No interpretability claim at all; **the learned adjacency is never
visualized or compared against Metr-La/Pems-Bay's actual known road/sensor
topology** — a free validation opportunity the paper does not take.
Efficiency claim (Fig. 6, relative time/epoch): MTGODE ×1.0 vs MTGNN ×1.34
vs GMAN ×28.08, obtained partly by "we slightly increase the spatial and
temporal step size... to trade model precision for speed" (p.9179) — an
explicit precision-for-speed trade acknowledged in the text, worth noting
when citing the efficiency number.

#### H. Writing
IEEE TKDE full-journal format: heavy theory-first structure (2 formal
Propositions with proofs deferred to online supplemental appendices),
explicit complexity analysis section (`O(Nd²+N²d)` for structure learning,
etc.) benchmarked directly against MTGNN's discrete complexity. Very short,
almost perfunctory Conclusion (one paragraph) with **no explicit
Limitations subsection** — the only concession is buried mid-section:
"we can find a sweet spot when selecting the spatial or temporal
integration time" (i.e., dataset-specific tuning with no principled
selection rule).

#### I. Red flags
Zero epidemiological content anywhere in 13 pages (no mention of Cola-GNN,
EpiGNN, STAN, or MPNN). Graph learner trains `2Nd+2d²` free parameters with
no supervision signal beyond the forecasting loss; on the 47-node,
348-week Japan-Prefectures scale this would almost certainly overfit given
the paper's own datasets are 50–5,000x longer.

#### J. Relevance
Reinforces the audit's verdict: architecturally portable (Eq. 6 accepts any
adjacency `Â`, so substituting a known geographic adjacency for the learned
one is a one-line change) but **statistically incompatible** with the scale
of the benchmark this project uses. The theoretical framing — decoupling
propagation *depth* from integration *time* to avoid over-smoothing by
construction, with a formal convergence proof — is a genuinely different
and citable argument from "we added residual connections," and is the kind
of rigor this project's own architecture-audit findings (E13, O(N²d) not
O(N)) would benefit from citing as a contrast: MTGODE proves a property
Wang's survey (paper 7 below) flags as merely empirically observed
elsewhere.

---

### 5. Panagopoulos, Nikolentzos & Vazirgiannis (2021), MPNN+TL

#### A. Bibliographic
*Proceedings of the AAAI Conference on Artificial Intelligence*
35(6):4838–4845, May 2021. École Polytechnique / AUEB. arXiv:2009.08388v5.
**Code: https://github.com/geopanag/pandemic_tgnn** (confirmed present,
footnote 4 on Dataset section).

#### B. Claimed contributions (verbatim)
> "We propose a model for learning the spreading of COVID-19 in a country's
> graph of regions. The model relies on the representational power of GNNs
> and their capability to encode the underpinnings of the epidemic. We
> apply a method based on MAML to transfer a disease spreading model from
> countries where the outbreak has been stabilized, to another country
> where the disease is at its early stages. We evaluate the proposed
> approach on data obtained from regions of 4 different countries... We
> observe that it can indeed surpass the benchmarks and produce useful
> predictions."

#### C. Method
**No epidemiological structure at all** — the "epidemiology" is metaphoric.
Graph = **daily Facebook Data-for-Good mobility matrix** between NUTS-3
regions (directed, weighted by device movement counts); node feature =
past `d` days of confirmed cases. **MPNN**: one message-passing layer per
snapshot, `H^{i+1}=f(ÂH^iW^{i+1})`, concatenated skip connections across
layers, ReLU output (nonnegative counts). **MPNN+LSTM**: sequence of daily
MPNN embeddings fed to a 2-layer LSTM. **MPNN+TL**: same MPNN base learner,
but meta-trained via **first-order MAML** across country×train-size×horizon
"tasks," Hessian term dropped (explicit statement that Finn et al. showed
it contributes little), then fully fine-tuned per target country. Loss:
plain MSE, Eq. 1.

#### D. Data & protocol
4 EU countries (Table 1): Italy 105 regions (24 Feb–12 May, avg 25.65
new cases/day), England 129 regions (13 Mar–12 May, avg 16.7),
Spain 35 regions (12 Mar–12 May, avg 61), France 81 regions (10 Mar–12
May, avg 7.5). Regions with <10 total confirmed cases dropped (14 Spain, 3
Italy). **Expanding-origin protocol**: T (train-set length) starts at 14
days, grows by 1 day per step; **a separate model is trained for every
(T, horizon-j) combination**, horizons j=1..14 days. Validation = days
T−1,T−3,T−5,T−7,T−9 (no held-out validation *window*, just 5 specific
lookback days). **No seed averaging anywhere — single run per
configuration.** Error metric (Eq. 5): mean absolute error per region,
averaged over the dt-day prediction window and all regions. Also a
region-relative error metric (Eq. 6) to control for population-size effects
between regions.

#### E. Baselines; re-run vs copied
All 8 baselines run by the authors on the same 4-country data: AVG,
AVG_WINDOW, LAST_DAY (true persistence), LSTM (Chimmula & Zhang 2020),
ARIMA (Kufel 2020), PROPHET (Mahmud 2020), **TL_BASE** (MPNN pretrained by
pooling all 3 other countries' data, then tested on the 4th — the critical
control for "is the gain from MAML, or just from more data?"), MPNN,
MPNN+LSTM.

#### F. Headline numbers (Table 2, verified against the PDF)
14-day-ahead avg error/region: England — AVG 10.09, LAST_DAY 8.66,
AVG_WINDOW 9.10, LSTM 7.02, ARIMA 15.65, PROPHET 16.24, TL_BASE 13.48, MPNN
8.13, MPNN+LSTM 6.93, **MPNN+TL 6.84**. France: AVG 8.55, LAST_DAY 7.24,
AVG_WINDOW 7.91, TL_BASE 12.27, MPNN 6.93, **MPNN+TL 6.13**. Italy: AVG
23.09, LAST_DAY 19.45, TL_BASE 24.89, MPNN 17.88, **MPNN+TL 16.69**. Spain:
AVG 47.63, LAST_DAY 42.79, TL_BASE 59.68, MPNN 35.31 [table: MPNN 35.83 at
3-day / for 14-day row MPNN=35.31], **MPNN+TL 34.65**. All numbers confirmed
matching the audit exactly. **TL_BASE (naive pooled pretraining) is worse
than MPNN alone in every one of the 12 country×horizon cells** — the
decisive evidence that the gain is attributable to MAML's per-task gradient
structure, not simply "more training data." At 3 days, MPNN+TL's margin
over AVG_WINDOW is only 3–8%, and LSTM/ARIMA/PROPHET **all underperform
LAST_DAY (persistence) in most cells** — this project's own baseline-hygiene
critique is directly borne out here, in the authors' own published table.

#### G. Ablations/interpretability
No formal ablation beyond the model-variant comparison (MPNN vs
MPNN+LSTM vs MPNN+TL) that is itself the main result table. No
interpretability claim — the message `Z_u = (x_j a_{j,u}+x_i a_{i,u}+
x_v a_{v,u})+x_u a_{u,u}` (Fig. 3) is offered as an intuitive read of the
architecture, not validated against anything external. Fig. 7 choropleth
maps show relative error vs average-case-count per region across all 4
countries — regions with the *fewest* cases have the highest relative
error, a sensible and explicitly discussed finding (small-count regions are
intrinsically noisier).

#### H. Writing
AAAI conference format (9 pages, dense, 2-column, no supplementary
appendix visible in what was read). Honest self-assessment, verbatim:
> "even though the MPNN+TL outperforms the baselines, their predictions
> are not very accurate in terms of average error." Followed immediately by
> a re-framing in terms of *relative* error (a region predicted 200 vs
> actual 160–240 is operationally fine; a region predicted 20 vs actual 10
> is 100% relative error but operationally acceptable) — a genuinely
> thoughtful discussion of how to read aggregate MAE in a policy context.

#### I. Red flags
The model *is* the mobility graph — swap in geographic adjacency and MPNN
degenerates to a vanilla GCN (its entire architectural motivation
disappears). No recovered/deaths data used by design (stated explicitly as
a data-availability limitation, not a methodological choice), which rules
out any SIR-style structure a priori — and the authors note they tried a
simple SI-with-published-β preliminary experiment and got errors "in a
different scale... which is why we have not experimented further" (a
second independent report of the same mechanistic-model failure mode Gao
et al. 2020/STAN also encountered). Single run per (T, horizon) cell — no
seed variance reported for any of the ~10 model configurations × 4
countries × 14 horizons.

#### J. Relevance
The single most **directly transportable protocol** in this reading group:
leave-one-country/dataset-out FOMAML transfer, with a naive-pooling control
(TL_BASE) that proves the mechanism isn't just data volume. Both this paper
and Nikparvar (below) explicitly name **cross-outbreak / cross-disease
transfer** as unexecuted future work — "Our final goal is to evaluate the
model on the second wave of COVID-19, based on the first" — still open as
of this reading. This is directly buildable on the six-dataset benchmark
already in this repo (counts + adjacency only, no mobility needed, since
the transfer target here is a graph-free common-factor model): leave one
region/season out, meta-train the seasonal-memory component across the
rest, fine-tune on the target — with TL_BASE-style naive pooling as the
mandatory control to isolate the transfer mechanism from raw data volume.

---

### 6. Wang, Y. (2023), *Advances in spatiotemporal graph neural network prediction research*

(File named `Jun Zhao et al_2023...pdf` — confirmed single-author paper,
see note at top.)

#### A. Bibliographic
*International Journal of Digital Earth* 16(1):2034–2066.
DOI 10.1080/17538947.2023.2220610 (a published correction exists at
10.1080/17538947.2023.2239591 — cite the corrected version). Yi Wang,
School of Earth and Space Sciences, Peking University. Open access, review
article. No code (survey).

#### B. Claimed contributions (verbatim)
> "Firstly, the definition of graph notation, the introduction of graph
> convolution, and the paradigm of ST-GNNs are comprehensively introduced.
> Secondly, 59 commonly used ST-GNNs frameworks, from the birth of ST-GNNs
> in 2017 to now, are reviewed. These frameworks with their datasets are
> summarized and classified from both the construction perspective and
> application perspective... Thirdly, the evolution history and future
> direction of the spatiotemporal graph convolutional prediction model are
> summarized and analyzed."

#### C. Method
Not a method paper — a taxonomy/survey. Three orthogonal classification
axes: **temporal** (RNN-based / CNN-based / Attention-based, minor
MLP/GCN-based classes), **spatial** (spectral-domain vs spatial-domain graph
convolution — **GAT is filed under spatial-domain, not as a separate
category**), and **graph type** (static / dynamic / multi-scale /
adaptive — confirmed the text states "It is divided into three categories"
immediately before listing four, the arithmetic error the earlier audit
flagged; verified present verbatim on p.2041). 59 models tabulated (Table
2) by presenting year, construction category, applying technology, and
graph type; a second table (Table 3) cross-references models against 3
data categories (dense/sparse/similar-dense) and named datasets.

#### D–F. Not applicable in the usual sense — a review paper.
Performance comparison tables are **copied from the cited papers, not
re-run**: "These results are cited from literature" (p.2049), and the
paper's own fairness bar is explicit: "If their evaluation results of
benchmarks are the same or less different (**no more than 5%**) from the
original paper, it indicates that their proposed model defaults to a fair
comparison." Table 4 (METR-LA, 15/30/60min RMSE/MAE/MAPE across 12 models)
and Table 5/6 (training time, inference time, parameter count, efficiency
tiers) are the paper's two quantitative deliverables, both dense-traffic
data only. **No significance testing, no seed/variance discussion, no split
protocol discussion anywhere in the paper** — confirmed absent throughout.

#### G. Ablations/interpretability
None (survey). Interpretability of attention is treated as self-evident:
"The attention mechanism obtains the input variables of the next layer by
weighting the hidden states of all-time steps of the encoder... express[ing]
the strength of information in the form of scores" — **no discussion of
attention-as-explanation faithfulness, no mention of graph-attention
validation against ground-truth structure, anywhere in the 34 pages.**

#### H. Writing
Standard T&F review format: Introduction → Paradigm/Definitions (§2, with a
clean symbol table, Table 1) → §2.3 Composition classification (model
taxonomy by construction) → §2.4 Data classification (application domain
taxonomy) → §2.5 Performance/efficiency comparison → §2.6 "Prediction
processes" (a 4-stage generic pipeline: preprocessing → construction →
training/adjustment → evaluation, with pseudocode, Algorithm 1) → §3
Evolution history and future directions (3 examples walked through in
detail: STGCN "complete correlation," ASTGCN "dynamic correlation," STSGCN
"synchronous correlation," each with an architecture figure and a
head-to-head performance comparison table) → §3.2 five explicit named
future-direction subsections → Conclusion. A large, well-organized
Appendix: Table A1 (abbreviation glossary for all 59+ models) and Table A2
(dataset name → public download link for every cited dataset) — genuinely
useful as a literature map, and a format worth imitating for a
related-work section that needs to cover many prior architectures compactly.

#### I. Red flags
The self-described "fair comparison ≤5% deviation" bar is generous and
unverifiable from outside (no access to the underlying re-derivation);
combined with "no more than 5%" being the *only* stated fairness criterion
and no discussion of seeds/splits, the cross-paper performance tables (4–6)
should be treated as indicative only, not citable point estimates.

#### J. Relevance
Confirms and extends the audit's central finding: only **2 of 59** models
are epidemic-related (CovidGNN, STAN — both listed with **no adaptive
adjacency, no multi-hop diffusion**), and there is **no epidemic accuracy
number anywhere in the survey**. The single most citable sentence for this
project's framing (§2.4.2, p.2045):
> "Sparse spatiotemporal graph data prediction mainly includes... crime case
> prediction, infectious disease prediction... their data values are zero in
> most cases... This type of model must be aware of the problems arising
> from zero-value inflation."
This independently corroborates (from a source with zero connection to this
project) that zero-inflation in sparse epidemic count data is a
named-but-unsolved problem in the wider ST-GNN literature — directly
supporting a paper section that frames graph-attention collapse and the
mismatch between dense-traffic-tuned architectures and sparse epidemic
counts as a structural, not incidental, failure. §3.2.3's verdict that
adaptive-adjacency learning is "**solved**" on dense data and merely
under-adopted (not an open research question) is exactly the framing this
project should use to justify *not* proposing yet another adaptive-graph
variant, and instead pivoting to a graph-free common-factor design.

---

### Synthesis (~300 words)

Across all seven papers, a single, load-bearing pattern recurs and is
directly corroborating for this project's Paper B: **wherever a paper
reports a component-removal ablation on a spatial/graph-attention branch,
the branch's measured value is small at short horizons and grows at long
horizons or regime changes** (Yin/STGNN's GAT-ablation loss grows from
4.7% at h=1 to 8.4–7.2% at h=5–7; Panagopoulos's MPNN+TL margin over
persistence-family baselines grows from 3–8% at 3 days to 12–22% at 14;
Jin's MTGODE ablations show the largest gains from CGP specifically at
longer propagation/integration depths). None of these seven papers, taken
individually, refutes graph attention — but none validates it as anything
beyond a mild, horizon-dependent correction either; interpretability claims
throughout the set (Kosma's "interpretable neural networks," Shi's LIME
importances, Yin's attention heatmaps) are asserted in the abstract and
supported by exactly one qualitative figure, never by a quantitative
faithfulness or ground-truth-recovery test — the same gap this project's
Paper B is built to close with a systematic collapse-to-uniform-pooling
demonstration. Baseline hygiene is uneven: Panagopoulos and Shi both
include real persistence baselines (and Panagopoulos's own table shows
LSTM/ARIMA/PROPHET losing to LAST_DAY in most cells — a striking
self-published admission); Yin's paper has none. Two protocols are worth
adopting directly: Panagopoulos's TL_BASE-controlled MAML transfer (proves
gains aren't just "more data") and Kosma's OOD-bin / cross-graph-size
generalization tests — both map cleanly onto testing a scale-equivariant
common-factor model across regions/populations never seen in training, not
just held-out time. Wang's survey independently names zero-inflation in
sparse epidemic counts as an unsolved, structural problem separate from the
dense-traffic literature its 59 models were built and tuned on — useful,
disinterested corroboration for framing why architectures validated on
METR-LA/PeMS-Bay transfer poorly to epidemic surveillance data.

---

## G6 — Surveys, agendas, probabilistic & physics-informed epidemic deep learning

> **Editor's note (29 Sep 2026).** Item 2's "Wang 2023" survey is Yi Wang (2023), *International Journal of Digital Earth* 16(1):2034–2066, read in full in G5.

Close-reading notes. Read via WebFetch of arXiv HTML (experimental full-text) and PMC full text where available.
All items read via automated full-text fetch/extraction (not manual PDF read), so quotes below are as
extracted by the fetch tool from the source HTML — treated as reliable but I flag anywhere content looked
truncated by the fetch.

---

### 1. Liu, Wan, Prakash, Lau, Jin — "A Review of Graph Neural Networks in Epidemic Modeling" (KDD 2024, arXiv:2403.19852)

**A. Bibliographic + code.** Zewen Liu, Guancheng Wan, B. Aditya Prakash, Max S. Y. Lau, Wei Jin. KDD 2024
(applied data science track survey). Curated paper list: github.com/Emory-Melody/awesome-epidemic-modeling-papers.
Same lab group later built EpiLearn (item 6) as the companion toolkit.

**B. Claims/key statements (quoted).**
- "GNNs stand out for their ability to aggregate diverse information through a message-passing mechanism,
  making them particularly suited [to epidemic modeling]."
- On mechanistic models: "they often suffer from limitations of oversimplified or fixed assumptions, which
  could cause sub-optimal predictive power."
- On hybrid models: "integration allows for the structured, theory-informed insights of mechanistic models
  to complement the flexible, data-driven nature [of neural models]."
- On real-world diffusion: "in the real world, disease spreading is a continuous process, which is
  incompatible with current methods."

**C. Method/scope.** A taxonomy paper, not a model. Splits the field by epidemiological *task* (Detection,
Surveillance, Prediction, Projection) and by *methodology* (Neural Models: spatial dynamics / temporal
dynamics / intervention modeling; Hybrid Models: GNN-assisted parameter estimation for SIR/SEIR/SIRD vs.
mechanistic-informed neural models). Positions itself as bridging the GNN and epidemiology communities.

**D. N/A (survey, not a trained model).** No standard benchmark table across surveyed papers; the survey
notes most work is COVID-19-only with a minority on ILI or other diseases, and does not tabulate
Japan-Prefectures/US-Regions/US-States protocols directly — that level of comparison is left to the
individual papers it cites.

**E/G-relevant: open problems (Section 5, quoted as extracted):**
- **5.1 Epidemic at Scales**: "existing approaches are limited to processing only two predefined scales,
  such as county-level and state-level data... there is growing anticipation for... models capable of
  incorporating data across multiple dynamic scales." Notes scalability tension: finer granularity blows up
  the graph while some tasks need real-time processing.
- **5.2 Cross-Modality**: "there has not been much work exploring the multi-modality of GNNs in an
  epidemiology setting."
- **5.3 Epidemic Diffusion Process**: continuous-time modeling gap (as quoted in B).
- **5.4 Interventions**: most work models only one intervention type (node- or edge-level), not both/multi-scale.
- **5.5 Generating Explainable Predictions**: "neural models investigated thus far have not placed
  significant emphasis on this aspect."
- **5.6 Data challenges**: noisy data ("the denoising mechanisms for GNNs in epidemiology have remained to
  be studied"), incomplete data, and privacy (proposes Federated Graph Learning).

**E. Attention — important negative finding for us.** The survey's treatment of graph/spatial attention
(GAT, Cola-GNN's "additive attention," RESEAT's continuously-updated attention matrix) is **uncritical and
positive** — attention is listed among "the advanced frontier of research on graph-structured data" with no
discussion of whether learned attention actually deviates from uniform, no mention of collapse/failure
modes, and no citation of any paper that audited attention entropy or interpretability empirically. This is
a real gap the survey does not flag as an open problem, even though it explicitly has a section on
"Generating Explainable Predictions" (5.5) that stops short of asking whether the *existing* explanatory
mechanism (attention) is trustworthy.

**F. Red flags.** (i) No critical evaluation-practice section — despite being a 2024 KDD survey, it does not
address leakage, horizon-scoring conventions, or statistical significance testing across the reviewed
literature; that gap is filled instead by Rodríguez et al. (item 3) and Bosse et al. (item 7). (ii) Positive-only
framing of attention is a citable gap for our attention-collapse finding. (iii) Companion toolkit EpiLearn
(item 6) reveals the group's own benchmark choices (Cola-GNN, EpiGNN, MepoGNN, DASTGN, STAN) — worth
cross-checking against our own baseline set.

**G. Relevance to our paper.** Strong citation for framing "the field has not audited whether learned
spatial attention actually does anything" — this survey treats attention as a solved, beneficial component,
which is the orthodoxy our attention-collapse finding (E3/E7 in the ledger) directly contradicts. Also useful
as the standard taxonomy reference (Neural vs. Hybrid) to locate MSAGAT-Net (a "Neural Model" combining
temporal conv + graph attention + multi-hop spatial conv) and to justify going graph-free as addressing
open problem 5.6 (noisy/incomplete data) — a graph-free common-factor model sidesteps the "denoising
mechanisms for GNNs... have remained to be studied" gap entirely by not requiring a graph.

---

### 2. "Wang et al. 2023" deep-learning-for-epidemic-forecasting survey — **NOT CONFIDENTLY IDENTIFIED**

I ran roughly a dozen targeted searches (exact-phrase search for "merely focus on proximity, yet ignore the
trend and periodicity"; author-name variants "Lijing Wang," "Yutong Wang," "Zhaoyang Wang," "Wang Xu";
venue-targeted searches across IEEE Access, Frontiers in Public Health, ACM Computing Surveys, Information
Fusion, Expert Systems with Applications; and searches combining "survey"/"review" + "epidemic forecasting"
+ "deep learning" + 2023) and could not locate a paper matching this description with confidence. Candidates
checked and ruled out:
- Liu, Liu, Liu — "Machine Learning for Infectious Disease Risk Prediction: A Survey" (arXiv:2308.03037,
  ACM Computing Surveys) — authors are Liu, not Wang; no confirmation of the quoted language.
- Yu, Xia, Li, Hou, Sheng — "Spatio-Temporal Graph Learning for Epidemic Prediction" (ACM TIST,
  10.1145/3579815) — a model paper (STEP), not a survey, and not authored by Wang.
- Lijing Wang et al.'s own model papers (CausalGNN AAAI'22, TDEFSI/DEFSI) are model papers, not surveys.
- I directly fetched the MPSTAN paper (arXiv:2306.12436), a plausible place for this exact critique sentence
  to appear as related-work framing, and it does not contain the quoted phrase either.

**I did not fabricate a match.** This item should be re-identified either from the paper's own reference
list (if the team has it) or by a differently-scoped search (e.g., Google Scholar direct, or Semantic
Scholar API with authentication, which I did not have access to in this tool set). Flagging for the team:
if you have the exact quoting paper (ours, or another lit-review reader) that cites this "Wang 2023" survey,
please share the DOI/arXiv ID and I (or another reader) can complete this entry.

---

### 3. Rodríguez, Kamarthi, Agarwal, Ho, Patel, Sapre, Prakash — "Data-Centric Epidemic Forecasting: A Survey" (arXiv:2207.09370)

**A. Bibliographic + code.** Alexander Rodríguez, Harshavardhan Kamarthi, Pulak Agarwal, Javen Ho, Mira
Patel, Suchet Sapre, B. Aditya Prakash. arXiv v2 HTML read in full. A version of this survey was later
published as "Machine learning for data-centric epidemic forecasting," *Nature Machine Intelligence* (2024)
— same lab/lineage; worth noting the NMI version exists as the "citable" venue.

**B. Claims (quoted).**
- Framing: "This survey delves into... data-driven computational methods, which have shown great potential
  in leveraging advances in data science and artificial intelligence and the incorporation of novel sources
  of information, from biological to behavioral."
- On metrics: "the selected metrics do not make the distinction in underestimating or overestimating, which
  has a different impact on real decision making."
- On real-time evaluation: "*Simulated real-time forecasting* setting... means using the version of the data
  available at a particular moment in time... enable modelers to train models with unrevised data and then
  evaluate them in curated data that best represents the actual burden of the disease," with the explicit
  warning that "models using revised data for training and evaluation may lead to different assessments."
- On short-termism: "most work in epidemic prediction has been focused on short-term predictions, typically
  up to 4 weeks in the future," questioning whether long-term epidemic forecasting is even possible given
  theoretical/practical limits.

**C. Method/scope.** Not a model paper. Organizes the field into three paradigms — Mechanistic
(compartmental/metapopulation/agent-based), Statistical/ML/AI (regression, neural sequence/similarity/
transfer/multimodal/spatial models, density estimation for UQ), and Hybrid (data assimilation, mechanistic
priors, ensembles/wisdom-of-crowds). Explicitly frames itself as bridging epidemiology and ML/data-science
communities, similar rhetorical move to item 1's survey but broader (covers statistical and mechanistic
work, not just GNNs).

**D. Evaluation practice — the most directly useful section for us.**
- Point-forecast metrics: MAE/RMSE/MAPE/NMSE used inconsistently across the field; task-dependent
  preference (MAE/RMSE for incidence and peak intensity; MAE for onset/peak-time event-type targets).
- Probabilistic scoring: documents CDC's adoption of a modified log score, `Log score(p,i) = ln(p_i)`, and
  flags unresolved sensitivity to the binning width used to discretize continuous forecasts into probability
  bins ("it is unclear what are the effects of the selection of [bin width]").
- Benchmarks referenced: FluSight (CDC, since 2013) and its cousins for Ebola/Dengue/COVID across ECDC,
  IARPA, PAHO; data sources CDC ILINet, FluSurv-NET/COVID-NET (14-state hospitalization panels), JHU
  COVID dashboard. Does *not* provide a systematic train/test-split table across reviewed papers (i.e., it
  does not itself audit evaluation protocol bugs the way Bosse et al. or a dedicated methods paper would) —
  its critique is about metric *choice* and real-time data-revision hazards, not about split/leakage bugs
  per se.

**Open problems (Section 9, quoted where possible; fetch truncated some subsections):**
1. Data-related challenges (reporting delay, right-truncation, representativeness bias in digital
   surveillance) — details truncated by the fetch tool but flagged as foundational throughout the survey.
2. Moving beyond short-term forecasting (quoted above).
3. Modeling multi-scale spatial-temporal dynamics — "spatial-temporal heterogeneity remains unresolved,
   particularly in incorporating sub-regional variation and multi-level contact networks."
4. Improving combination of models / wisdom-of-crowds — ensembling lacks principled theory.
5. **Providing well-calibrated and explainable forecasts** — explicitly names calibration and UQ,
   "particularly in deep learning approaches," as unresolved.
6. Setup and evaluation for *actionable* forecasts — gap between statistical accuracy metrics and
   decision-maker utility.
7. Technical debt of real-world deployment (pipeline/versioning/maintenance).

**E. Writing/structure.** Long, taxonomic survey with a pipeline figure (data → paradigm → evaluation).
Explicitly non-adversarial about mechanistic vs. ML approaches — treats hybrid methods as the frontier. Style
is closer to a roadmap/agenda paper than a narrow technical review; heavy on "what's missing" framing
throughout, not just in the final section.

**F. Red flags.** The survey's own spatial/graph-modeling coverage is thin (their neural-models subsection
6.3.5 on "incorporating spatial structure" is not developed in depth in the fetched excerpt) — they *name*
multi-scale spatial modeling as an open problem (challenge 3) but do not audit whether the graph structures
already in use (adjacency, mobility) are doing real work, which is exactly the gap our attention-collapse
finding addresses empirically rather than programmatically.

**G. Relevance.** This is probably our single best "field says calibration/UQ and actionable-evaluation are
unsolved" citation (challenge 5 and 6 map directly onto our calibration + power-analysis + DM-test +
per-region conformal design). Their real-time-data-revision warning is directly relevant to any claim we
make about protocol correctness. Their point that spatial-temporal heterogeneity/multi-scale dynamics is
unresolved (challenge 3) is a problem our graph-free common-factor design *sidesteps* rather than solves —
worth being precise about that distinction in the manuscript (we're not solving multi-scale graph modeling,
we're arguing the graph wasn't earning its keep at the scale tested).

---

### 4. Kamarthi, Kong, Rodríguez, Zhang, Prakash — CAMul: "Calibrated and Accurate Multi-view Time-Series Forecasting" (WWW 2022, arXiv:2109.07438)

**A. Bibliographic + code.** Harshavardhan Kamarthi, Lingkai Kong, Alexander Rodríguez, Chao Zhang,
B. Aditya Prakash. WWW 2022 (Web Conference). Same lab lineage as items 3, 5.

**B. Claims (quoted).**
- "CAMul outperforms other state-of-art probabilistic forecasting models by over 25% in accuracy and
  calibration."
- Mechanism framing: "integrating information and uncertainty from each data view in a dynamic
  context-specific manner."
- "View selection module, indeed, selects the most informative views on average" — i.e., they claim their
  attention/weighting module is doing real selective work (a claim category directly analogous to what we
  are refuting for MSAGAT-Net's spatial attention — worth contrasting rhetorically: did CAMul ever validate
  this claim with an entropy/ablation audit, or is it asserted from aggregate accuracy improvement alone?
  Based on the fetched text, the claim is supported by held-out accuracy, not by an independent
  interpretability audit — i.e., the same evidentiary gap we're flagging elsewhere in the field).

**C. Method (5-8 lines).** CAMul is a multi-view probabilistic forecaster: each data modality (case-count
sequences via GRU, graph-structured features via GCN, static features via feed-forward) is encoded into a
*distribution* over a latent embedding, not a point embedding. A "View-Specific Correlation Graph" built
from RBF similarity in latent space captures view-aware uncertainty; a cross-attention module dynamically
reweights view importance conditioned on the input sequence; weighted embeddings feed a decoder producing
Gaussian mean/variance for a calibrated predictive distribution.

**D. Data/protocol/numbers.** No Japan-Prefectures/US-Regions/US-States benchmark — CAMul's benchmarks are:
(1) google-symptoms — flu, 8 US HHS regions, Google symptom search data (2017+), 1–4 week horizons; (2)
covid19 — mortality, 50 US states, June–Dec 2020, 1–4 weeks; (3) power — electricity, 1-min-ahead (non-epi);
(4) tweet — COVID-topic tweet distributions, 50 states, 15 weeks. Protocol: 10% validation split, 20
experimental runs, t-test at α=1% for significance. Headline numbers (as extracted): covid19 RMSE 27.3 vs.
CMU-TS 32.4, CRPS 23.8 vs. 30.1; google-symptoms RMSE 0.49 vs. EpiFNP 0.64. Calibration measured via a
**Confidence Score** `CS(M) = ∫₀¹ |k_M(c) − c| dc` (deviation of empirical coverage from nominal confidence,
0 = perfectly calibrated) plus CRPS and interval score.

**E. Writing/structure.** Standard ML-conference model paper: motivation (multi-source epidemic signals are
heterogeneous and noisy), method, four benchmarks, ablations, calibration analysis. Explicit emphasis on
calibration as a first-class metric alongside accuracy, unusually so for this literature at the time (2021-22).

**F. Red flags.** Uses HHS-region/state granularity, not the Japan-Prefectures/US-Regions/US-States trio our
paper and its direct competitors (Cola-GNN/EpiGNN) use — so it is not a *baseline* we can directly compare
against on our exact splits, only a citation for calibration methodology and for the general finding that
multi-view/uncertainty-aware epidemic forecasters were already claiming large calibration gains by 2022. The
"view selection... selects informative views" claim is asserted, not independently audited — same
evidentiary gap pattern we're calling out for spatial attention elsewhere.

**G. Relevance.** Best available citation for a formal, per-forecast **calibration score definition**
(the confidence-score integral) that we could cite or adapt for our own per-region conformal
interval reporting, and for establishing that "calibration" has been a recognized first-class evaluation
axis in this exact sub-community (Prakash lab) since 2022 — i.e., a 2026 reviewer from that community will
expect us to report something CAMul-comparable, not just point accuracy.

---

### 5. Rodríguez, Cui, Ramakrishnan, Adhikari, Prakash — EINNs: "Epidemiologically-Informed Neural Networks" (AAAI 2023, arXiv:2202.10446)

**A. Bibliographic + code.** Alexander Rodríguez, Jiaming Cui, Naren Ramakrishnan, Bijaya Adhikari,
B. Aditya Prakash. AAAI 2023.

**B. Claims (quoted).**
- "we do not assume the observability of complete dynamics and do not need to numerically solve the ODE
  equations during training."
- "EINNs is the only one consistently providing accurate and well-calibrated forecasts."
- "predictions well-correlated with epidemic trends and long-term predictions remain open challenges" —
  stated as a residual problem even after their method.

**C. Method (5-8 lines).** Physics-informed hybrid: a "time module" (PINN) learns latent epidemic dynamics
by minimizing the residual between ODE-implied gradients (SEIRM for COVID, SIRS for flu) and neural-network
gradients via autodiff; a "feature module" (RNN) ingests multivariate exogenous features (mobility,
symptom surveys, hospitalization); knowledge transfers between modules via a "gradient trick" that matches
internal embeddings (e_t ≈ e_t^f) rather than requiring explicit ODE integration at every step — i.e., a
physics-informed regularizer applied through representation alignment, not loss-term ODE penalties alone.

**D. Data/protocol/numbers.** COVID-19: 47 US states + national; influenza: 10 HHS regions. Real-time
forecasting setup — trained only on data available as of the forecast date. Evaluation windows: COVID Sept
2020–Mar 2021 (8 months), flu Dec 2017–May 2018 (5 months); predictions issued every 2 weeks for 1–8-week
horizons, "5696 predictions per model across ~700 training runs." Baselines: naive RNN, standalone
SEIRM/SIRS, synthetic-data generation and ODE-regularization variants, ensembling, ablations. **No explicit
significance testing and no multi-seed/CV protocol reported** in the fetched text — a real methodological
gap relative to our power-analysis/DM-test standard. Headline numbers (normalized-error metrics NR1):
COVID short-term (1-4wk) EINNs 0.54 vs RNN 1.09, SEIRM 2.35; long-term (5-8wk) EINNs 0.85 vs RNN 1.19, SEIRM
7.14; Pearson r 0.46-0.53 range depending on horizon. No Japan-Prefectures/US-Regions/US-States (Cola-GNN
lineage) numbers — different benchmark family entirely.

**E. Scale/log-space.** **No log-space or scale-invariant target modeling** in this paper — targets are raw
mortality counts or an indirectly-computed ILI proxy (β·I·S/(N·OR)). They handle zero-count instability by
adding "+1 death" to denominators rather than a systematic log/Box-Cox transform. This is a clean, citable
gap: EINNs is a strong physics-informed baseline in spirit but does not do what Bosse et al. (item 7)
recommend (evaluate/model on a variance-stabilizing transformed scale), and does not use log-growth targets
the way our proposed design does.

**F. Red flags.** No reported significance testing; scale handling is ad hoc (add-one smoothing) rather than
principled; "well-calibrated" is asserted in the abstract-level claim but the calibration metric/definition
used to support it is not detailed in the extracted material — would need the full PDF to check whether
"well-calibrated" is backed by a coverage-based metric like CAMul's confidence score or is a looser claim.

**G. Relevance.** Useful physics-informed/hybrid comparison point: shows the field's most-cited
epidemiologically-informed neural model (i) still models on raw/near-raw scale rather than log-growth, (ii)
does not report significance testing, and (iii) claims calibration without (as far as extracted)
demonstrating it via an explicit coverage metric. All three are exactly the gaps our design (log-growth
targets, power analysis + DM tests, per-region conformal intervals) is built to close, so this is a good
"here is where the physics-informed sub-literature currently stands" anchor citation.

---

### 6. Liu, Li, Wei, Wan, Lau, Jin — EpiLearn (2024, arXiv:2406.06016; github.com/Emory-Melody/EpiLearn)

**A. Bibliographic + code.** Zewen Liu, Yunxiao Li, Mingyang Wei, Guancheng Wan, Max S.Y. Lau, Wei Jin —
same lab as item 1's KDD 2024 survey. Live GitHub repo (actively maintained; repo state as of this read is
ahead of the original 2024 arXiv paper).

**B. Claims (quoted).**
- "EpiLearn not only provides support for evaluating epidemic models based on machine learning, but also
  incorporates comprehensive tools for analyzing epidemic data."
- "there is a growing need for a unified platform to facilitate the study and development of computational
  methods for epidemic modeling" — explicit motivation that existing packages "have discontinued
  maintenance or failed to extend beyond traditional mechanistic models."

**C. Scope (5-8 lines).** A PyTorch/PyG-based benchmarking library, not a model paper. Modular design:
separate Dataset / Transform / Model / Task objects composed through a pipeline API, supporting two tasks
(Forecasting, Source Detection) in the original paper; the live repo has since expanded to four tasks
(Forecasting, Nowcasting, Scenario modeling, Source detection).

**Models (paper, original 2024 list):** Temporal — ARIMA, XGBoost, SIR, SIS, SEIR, GRU, LSTM, DLinear, EINN
(item 5's method is literally included as a baseline); Spatial — GCN, GAT, GIN; Spatio-temporal baselines —
CNNRNN, DCRNN, ST-GCN, GraphWaveNet; Spatio-temporal *epidemic-specific* models — **Cola-GNN, STAN, MepoGNN,
EpiGNN, EpiColaGNN, DASTGN**. The live repo now advertises "65 models spanning mechanistic, statistical, deep
temporal, spatiotemporal graph, and time-series foundation models" — a much larger, continuously updated set.

**Datasets — direct check against our benchmark family.** The repo's `datasets/` directory (checked
directly) contains: `JHU_covid.pt`, `Tycho_v1.pt`, `benchmark.pt`, `covid_dynamic.pt`, `covid_static.pt`,
`measles.pt`, plus toy files. Documented sources: Project Tycho v1.0.0 (8 diseases, US states/cities,
1916-2009), Measles (England & Wales urban centers, 1944-1964), multi-country COVID (Brazil, China, Austria,
England, France, Italy, New Zealand, Spain). **No Japan-Prefectures, US-Regions, or US-States (Cola-GNN/
EpiGNN-style ILI benchmark) dataset files are present** in the repo as checked, despite Cola-GNN and EpiGNN
themselves being included as *models*. This is a genuine, checkable finding: EpiLearn standardizes the
model zoo but does **not** standardize on the exact benchmark family (Japan-Prefectures/US-Regions/
US-States) our paper and its direct baselines use.

**D. Protocol/scoring.** Lookback/horizon windowing (`lookback=12, horizon=3` in the documented example).
Per the live repo: "rolling-window evaluation with uncertainty," walk-forward validation folds, task-specific
metrics (MAE named explicitly for nowcasting), config-driven YAML benchmarking, and output of "per-model
metrics, conformal coverage, Optuna trials and raw predictions." **Notably: the live repo bakes conformal
prediction intervals into its standard evaluation output for every task** — a strong, independent signal
that conformal calibration is becoming an expected default in this sub-field's tooling, not just an
occasional add-on paper.

**E. Writing/structure.** Standard toolkit/resource paper: motivation (fragmentation), library architecture,
model/dataset catalog, case-study benchmarking, community/maintenance pitch (interactive web app for
visualization).

**F. Red flags.** (i) Benchmark-dataset coverage does not match the Japan-Prefectures/US-Regions/US-States
family used by our direct competitors, so we cannot lean on EpiLearn for apples-to-apples numbers against
MSAGAT-Net; would need to verify with the team whether the live repo has since added these (worth a repo
search closer to submission time, since it's actively developed). (ii) The two fetches (arXiv paper vs. live
repo) disagree substantially on model count (6 epidemic-specific spatio-temporal models in the paper vs. "65
models" claimed live) — the repo has clearly grown well past the archived paper, so any citation of
"EpiLearn supports N models" needs a version/date caveat.

**G. Relevance.** Two uses: (1) as evidence the field is converging on Cola-GNN/EpiGNN/MepoGNN/STAN as the
de facto standard spatio-temporal epidemic-GNN baseline set (useful to justify our baseline choices being
representative); (2) as evidence that **conformal prediction intervals are now considered baseline
infrastructure** for epidemic-forecasting evaluation tooling in 2024-2026, which directly supports our use of
per-region conformal intervals as an expected-not-novel piece of a rigorous 2026 evaluation protocol.

---

### 7. Bosse, Abbott, Cori, van Leeuwen, Bracher, Funk — "Scoring epidemiological forecasts on transformed scales" (PLOS Comput Biol 19(8):e1011393, 2023)

**A. Bibliographic + code.** Nikos I. Bosse, Sam Abbott, Anne Cori, Edwin van Leeuwen, Johannes Bracher,
Sebastian Funk. PLOS Computational Biology, published 29 Aug 2023. DOI 10.1371/journal.pcbi.1011393. Full
text read via PMC10495027.

**B. Claims (quoted).**
- Core problem: "applying these scores directly to predicted and observed incidence counts may not be the
  most appropriate due to the exponential nature of epidemic processes and the varying magnitudes of
  observed values across space and time."
- Asymmetry: missing a growth rate by ±ε produces an absolute-error ratio of exp(−εt) < 1 on the natural
  scale, i.e., **underprediction is systematically penalized less than equivalent overprediction** when
  scored on raw counts — this is a load-bearing, quantitative claim for anyone (us) arguing that raw-scale
  scoring biases evaluation.
- "Rankings between different forecasters based on the CRPS may change when making use of a transformation,
  both in terms of aggregate and individual scores" — i.e., which model "wins" is not scale-invariant for
  proper scores in general.
- Recommendation: "such evaluations on the logarithmic scale should complement the prevailing evaluations on
  the natural scale."
- Practical offset rule for log(x+a) with zero-handling: choose *a* such that x > 5a; **a = 1 recommended as
  suitable for most epidemiological targets.**
- Key exception: "The logarithmic score has scale invariance properties which imply that score differences
  between different forecasts are invariant to strictly monotonic transformations... The question of the
  right scale to evaluate forecasts on does therefore not arise for the log score" — i.e., proper log-scoring
  rules sidestep the whole problem, but CRPS/WIS (the field's actual default) do not.
- Mean-variance justification for log transform: empirically the mean-variance relationship in COVID
  incidence data is "somewhat below 1, implying a slightly sub-quadratic mean-variance relationship,"
  supporting (not exactly requiring) log transformation as variance-stabilizing (Poisson-like linear
  mean-variance would instead favor sqrt).

**C. Method/scope (5-8 lines).** Not a model paper — a methodological/statistical paper on scoring rule
theory as applied to epidemic forecasts. Establishes that any strictly monotonic transformation applied
identically to forecast and observation preserves propriety of the scoring rule; works through three
interpretations of log-CRPS (relative error, growth-rate scoring, variance stabilization); empirically
demonstrates rank changes using the European COVID-19 Forecast Hub (32 locations, 7 models, cases and
deaths, Mar 2021-Dec 2022, 1-4wk horizons).

**D. Protocol (as a methods paper).** European COVID-19 Forecast Hub data, log-transform with offset a=1,
comparing natural-scale vs. log-scale CRPS/WIS rankings; regression of log-variance on log-mean to estimate
the mean-variance exponent empirically per target/location.

**E. Writing/structure.** Theory section → practical considerations (offset/zero handling) → empirical
worked example (Forecast Hub) → regression analysis. Framing is careful and technical (biostatistics
audience), explicitly positions itself as a complement to, not a replacement for, natural-scale scoring.

**F. Red flags.** None substantive — this is a rigorous, narrowly-scoped statistics paper from a
well-regarded epidemiological forecasting group (epiforecasts.io / LSHTM). The main caveat for us: their
worked example is CRPS/WIS on aggregate case/death counts, not on log-*growth* targets specifically (growth
rate is discussed as a *motivating* asymmetry example, not as the primary scored quantity in their case
study) — so we should be precise that they justify log-scale evaluation of *incidence*, and we should not
overclaim that they validate log-*growth*-target modeling per se, though the argument transfers directly.

**G. Relevance — probably our single most load-bearing citation for the "log-growth targets" design
choice.** This paper is the standard 2023 reference a 2026 reviewer in this space will expect us to cite
the moment we say we score/model on log or growth-rate scale. Concretely: (i) it gives us the formal
justification (variance-stabilization + relative-error argument) for choosing log-growth targets, (ii) it
gives us the citable warning that raw-scale CRPS/WIS asymmetrically under-penalizes underprediction — which
we should use if our corrected-protocol paper discusses why prior raw-scale comparisons may have been overly
forgiving of models that systematically underpredict growth, (iii) it explicitly recommends *complementing*
natural-scale with log-scale evaluation rather than replacing it, which is a methodological posture we should
mirror rather than presenting log-growth scoring as strictly superior.

---

### 8. Conformal prediction for epidemic forecasting (2023-2026 sweep)

I ran an extensive search (FluSight-specific, spatio-temporal-specific, 2024/2025/2026-dated) and did not
find a well-established, heavily-cited paper doing *spatio-temporal* conformal prediction specifically for
multi-region epidemic forecasting in the 2024-2026 window. The two most relevant items found:

#### 8a. Susmann, Chambaz, Josse — "AdaptiveConformal: An R Package for Adaptive Conformal Inference" (arXiv:2312.00448, Dec 2023)

**A.** Herbert Susmann (CEREMADE), Antoine Chambaz (MAP5), Julie Josse (PREMEDICAL). arXiv stat.CO,
Dec 2023 — just outside the strict 2024-2026 window but the closest confirmed match to "adaptive conformal
for FluSight."

**B/C.** Implements five Adaptive Conformal Inference (ACI) algorithms as a black-box wrapper around
*existing point forecasts* — does not require exchangeability, adapts interval width online as data arrive.
Includes "a case study of producing prediction intervals for influenza incidence in the United States based
on black-box point forecasts," i.e., a FluSight-style US-influenza application, using ACI to retrofit
calibrated intervals onto arbitrary point-forecast pipelines.

**D.** Package/methods paper, not a benchmarked model paper with fixed splits/seeds in the extracted
material — the value here is methodological (the ACI algorithm family and its epidemic case study), not a
head-to-head accuracy table.

**G.** Directly relevant as the citable "conformal wrapper for epidemic point forecasts" precedent — supports
using conformal methods post-hoc around a trained forecaster (as opposed to requiring a purpose-built
probabilistic architecture like CAMul), which is closer to what a "per-region conformal interval" addition to
an existing point-forecasting model (ours) would look like.

#### 8b. Field-level signal from EpiLearn (item 6) and the FluSight ensemble coverage literature

The EpiLearn toolkit (item 6) now ships conformal-interval computation as a default output for every
supported task — independent evidence that conformal calibration is becoming expected tooling, not a novel
contribution, by 2024-2026. Separately, FluSight-ensemble coverage-decay statistics reported in the
literature (2021-22 season: 95% PI coverage 89.6%→83.7% as horizon grows 1→4 weeks; 2022-23: 85.7%→77.9%,
with coverage dropping below 50% during rapid growth phases such as Omicron) are widely cited as the
motivating empirical failure mode that adaptive/conformal methods are meant to fix — i.e., **standard
intervals under-cover specifically during the regime (rapid growth) our log-growth reframing targets**,
which is a nice rhetorical tie-in.

**F. Red flags for item 8 generally.** I could not verify a dedicated, well-cited 2024-2026 paper doing
region-wise (spatially disaggregated) conformal prediction for epidemic case counts specifically — if the
team's other readers or later searches turn one up, it should supersede 8a as the primary citation. Do not
cite anything beyond 8a/8b for this item without further verification; I am flagging incompleteness rather
than guessing.

---

### Synthesis (~400 words)

Read together, this group of eight sources maps three separate "field expects X by 2026" pressures our
design should explicitly answer, plus one clear gap we can exploit rhetorically.

**Evaluation protocol.** Rodríguez et al. (item 3) and the GNN survey (item 1) both name calibration/
uncertainty quantification and actionable, decision-relevant evaluation as unresolved (Rodríguez's
challenges 5-6); Bosse et al. (item 7) give the formal statistical reason raw-scale CRPS/WIS scoring is
asymmetric and rank-unstable, and explicitly recommend log-scale evaluation as a *complement* to, not
replacement for, natural-scale scoring. None of the model papers we read (CAMul, EINNs) actually adopt
log-growth targets — EINNs in particular still models on near-raw scale with ad hoc "+1" zero-handling. This
is the gap our log-growth, scale-equivariant design fills, and Bosse et al. is the citation a 2026 reviewer
will expect to see the moment we justify that choice.

**Calibration is now a checked box, not a novelty.** CAMul (2022) already treats calibration as a first-class
metric with a formal coverage-deviation score; EpiLearn's live tooling (2024-2026) bakes conformal intervals
into every task's default output; FluSight's own documented coverage decay (worse at longer horizons and
during rapid-growth phases — exactly where log-growth reframing should help most) is the field's standard
motivating failure case. A reviewer will expect calibration reported via a coverage-based metric (à la
CAMul's confidence score) or conformal intervals, not accuracy alone — our per-region conformal design
directly answers this, and 8a (AdaptiveConformal) is the right methodological precedent for a "wrap
calibrated intervals around a point forecaster" design rather than requiring a from-scratch probabilistic
architecture.

**Attention orthodoxy is under-scrutinized.** The KDD 2024 GNN survey (item 1) treats attention (GAT,
Cola-GNN's additive attention, RESEAT) uncritically as an unqualified positive, with no discussion of
entropy collapse or whether it does anything beyond weighted pooling — despite having a dedicated
"explainability" open-problem section. This is a genuine, citable orthodoxy gap that our attention-collapse
finding contradicts empirically; no source in this group offers a competing empirical audit, which makes our
finding more novel, not less.

**Key citations to carry forward:** Bosse et al. 2023 (log-scale scoring justification — load-bearing), item
1 and item 3 surveys (open-problems framing, especially calibration/UQ and explainability gaps), CAMul
(calibration-score precedent), EpiLearn (benchmark/tooling convergence + conformal-as-default signal), and
8a AdaptiveConformal (conformal-wrapper precedent). Item 2 (Wang 2023) remains unresolved and should be
chased down by title/DOI before the manuscript cites it secondhand.

---
