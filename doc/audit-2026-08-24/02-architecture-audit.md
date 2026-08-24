# 02 — Architecture audit

Read 24 Aug 2026: `src/models.py` (1084 lines), `src/train.py` (766),
`src/data.py` (303), `src/evaluate.py` (424), `src/utils.py` (633), line by
line, with the parameter arithmetic verified against three trained
checkpoints (exact match) and the manuscript Methods section cross-checked.

---

## 1. Module-by-module forward pass

Entry point: `MSAGATNet_Ablation.forward` — `src/models.py:943-1046`.
Input `x: [B, T=window=20, N]` (min–max normalised), `x_last = x[:, -1, :]`
(`models.py:952`).

Defaults (`models.py:27-35`, `train.py:489-493`): `H=hidden_dim=32`,
`heads=4`, `head_dim=8`, `b=bottleneck_dim=8`, `d_feat=FEATURE_CHANNELS=16`,
`k=3`, `w=20`, `dropout=0.2`, `highway_window=4`.

### 1.1 TFEM — `DepthwiseSeparableConv1D` (`models.py:99-141`), instantiated `:723-729`

`x → [B*N, 1, T]` (`:955`) → depthwise `Conv1d(1,1,k=3,pad=1,dilation=1,groups=1)`
→ `BatchNorm1d(1)` → ReLU → pointwise `Conv1d(1,16,k=1)` → `BatchNorm1d(16)`
→ ReLU → Dropout → `[B*N,16,T]` → `view(B,N,320)` (`:958-959`).

**There is no multi-scale temporal convolution and no dilation.** `dilation`
is a constructor default of 1 (`:118,123`) and is never passed a value ≠ 1
anywhere in `src/`. The "Multi-Scale" in the model name refers exclusively
to *spatial hops* (MSSFM). The Related Work section attributes dilated
multi-scale temporal convolutions to Cola-GNN
(`doc/elservier/elsarticle-template-num.tex:265`), and the research ledger
confirms the dilated module ("MRTM") was never integrated.

**Degenerate depthwise-separable.** With `in_channels=1` the depthwise conv
is a single 3-tap FIR filter (3 weights + 1 bias). The pointwise conv is
`out[c,t] = w_c·z_t + β_c`; `BatchNorm1d(16)` then standardises each channel
over `(B·N, T)`, which cancels `w_c` up to sign. So the 16 "feature
channels" are exactly `ReLU(±γ_c·ẑ_t + β_c)` — sixteen thresholded copies of
**one** scalar filtered sequence. Effective temporal receptive field = 3
steps out of a 20-step window; everything longer-range must be recovered by
the flatten+linear.

Params: 3+1 (dw) + 2 (bn1) + 16+16 (pw) + 32 (bn2) = **70**.

### 1.2 Feature projection (`models.py:731-736`, forward `:962-965`)

`Linear(320→8)` → `Linear(8→32)` → `LayerNorm(32)` → ReLU. No nonlinearity
between the two linears, so this is exactly a **rank-8 linear map**
R^320 → R^32. Params: 2568 + 288 + 64 = **2920**. Output `F: [B,N,32]`.

### 1.3 EAGAM — `SpatialAttentionModule` (`models.py:144-372`), instantiated `:744-753`

Forward `:271-372`:

```
qkv = W_high(W_low(F))                          # 32 → 24 → 96, :288-289
q,k,v : [B,4,N,8]                               # :293
S      = q kᵀ / sqrt(8)                         # :296   content term
Bias   = U @ V                    [4,N,N]       # :299   learnable graph bias
S      = S + Bias                               # :306   (additive branch)
   or  S = S * sigmoid(Bias)                    # :304   ('multgate')
Ã      = A / rowsum(A)            [N,N] buffer  # :243-244
  ('adjstd')  Ã ← (Ã − rowmean)/(rowstd+1e-8)   # :320-322
S      = S + softplus(adj_scale) · Ã            # :312,323
  ('scorenorm') S ← (S−rowmean)/(rowstd+1e-8)   # :328-329
  ('temp')      S ← S · exp(log_attn_temp)      # :334
A      = softmax(S, dim=-1)                     # :337   stored on self.attn
O      = Dropout(A) @ v                         # :338-339
O      = W_out_high(W_out_low(concat_heads(O))) # :366-367  (rank-8 linear)
out    = LayerNorm(O + F)                       # :370
```

- **Heads** = 4, head_dim = 8 (`:179`).
- **Rank of the U@V bias** = `bottleneck_dim` = 8 by default, overridden by a
  `rankN` token (`:216-222`). `u: [4,N,r]`, `v: [4,r,N]`, both Xavier-uniform
  (`:223-226`).
- **`adj_scale`** is a scalar `Parameter(1.0)` (`:247`), passed through
  `softplus` → 1.313 at init. Note `softplus(0) = ln2 = 0.693`, so even when
  weight decay drives `adj_scale → 0` the prior does **not** switch off.
- **`adj_prior`** is a non-trainable buffer, row-normalised without an added
  identity — but every shipped adjacency already has a unit diagonal
  (verified: `diag_mean = 1.00` for all six matrices).
- The QKV "low-rank bottleneck" (32→24→96) is rank-24 of a possible 32 and
  costs **3192 params vs 3168** for a plain `Linear(32,96)` — i.e. the
  bottleneck *increases* the parameter count. The claim at
  `elsarticle-template-num.tex:606` that it reduces params from O(3d²) to
  O(3d·d_b) does not hold at d=32, d_b=8, 3d_b=24. The *output* bottleneck
  (32→8→32, 552 vs 1056) does save.

Params: `1 (log_attention_reg_weight) + 792 + 2400 + 264 + 288 + 64 (LN) +
1 (adj_scale) + 64N` = **3810 + 64N** (+1 with `temp`/`attn_fix`).

Memory note: `self.attn` retains `[B,4,N,N]`. LTLA at B=32: 17.7M floats
≈ 71 MB, held on the module and inside the autograd graph.

### 1.4 MSSFM — `MultiScaleSpatialModule` (`models.py:375-496`), instantiated `:761-767`

Hop matrices precomputed as buffers (`:437-462`): `Â = D⁻¹(A+I)`;
`adj_hop_0 = I`, `adj_hop_k = Â^k` for k=1..S−1.

`S = min(4, max(2, N//5))` (`:413`) → **S=4** for LTLA(372)/Japan(47)/States(49);
**S=2** for NHS(7)/Australia(8)/Regions(10).

```
H_k     = [Linear(32,32) → LayerNorm → ReLU → Dropout](Â^k G)   # :477-482
α       = softmax(fusion_weight)                                 # :485
H_fused = Σ_k α_k H_k                                            # :489
out     = LayerNorm( W_hi(W_lo(H_fused)) + G )                   # :492-494
```

`fusion_weight` init `exp(−0.5k)` (`:426-427`).
Params: `1121·S + 616`. S=4 → **5100**; S=2 → **2858**.

### 1.5 Optional spatial gate (`models.py:798-801`, forward `:974-976`)

`fusion = σ(g)·fusion + (1−σ(g))·features`, `g` init 0 → 0.5. Off by default.

### 1.6 PPRM — `HorizonPredictor` (`models.py:499-572`), instantiated `:778-783`

```
P_init = Linear(8,h)( Dropout(ReLU(LayerNorm( Linear(32,8)(H) ))) )   # :552-554
R      = σ( Linear(8,h)( ReLU( Linear(32,8)(H) ) ) )                  # :559
T      = x_last ⊙ exp(−exp(log_decay) · [1..h])                       # :563-565
P      = R ⊙ P_init + (1−R) ⊙ T                                       # :568
```

`log_decay` init −2.3 → γ≈0.1 (`:533`). Output transposed to `[B,h,N]`
(`:980`). Params: `264 + 16 + 9h + 1 + 264 + 9h` = **545 + 18h**.

### 1.7 Highway / AR (`models.py:786-791`, forward `:983-992`)

`z = Linear(4→h)(x[:, −4:, :])`;
`pred = σ(highway_ratio)·P + (1−σ(highway_ratio))·z`.
`highway_ratio` init 0.5 → σ = 0.622 (not 0.5; the manuscript at `.tex:1321`
says λ is "initialised to 0.5", which is the pre-sigmoid value).
Params: `5h + 1`.

### 1.8 Output heads

**(a) Direct head** — the highway-blended `predictions: [B,h,N]`. Under
`--target_space level` this is the normalised level; under `loggrowth` it is
`ĝ`.

**(b) Quantile head** — `QuantileHead` (`models.py:42-85`), attached
`:806-811`, called `:1042-1044`. Takes `fusion_features` (post-MSSFM,
pre-highway) and `point = predictions[:, −1, :]`:

```
inc  = softplus(Linear(16,Q−1)(Dropout(ReLU(Linear(32,16)(feat)))))
low  = reverse-cumsum(inc[..., :m]);  q_low = point − low
q_med= point
q_hi = point + cumsum(inc[..., m:])
```

Monotone by construction; the median is tied to the point forecast
(`:77-84`). With 23 levels, m=11, Q−1=22. Params **902** (+23 float buffer).

**(c) Renewal decoder** (`models.py:825-907`, forward `:996-1040`, coupling
`:1048-1072`). Off by default.

```
raw   = clamp(x·(dat_max−dat_min) + dat_min, 0)          # :1002-1003 denormalise
W     = softmax( mean_heads(U@V) + softplus(adj_scale)·std_row(Ã), −1 )   # :1055-1072
mixed = raw @ Wᵀ                                          # :1005
α     = softmax(log_alpha)                [L]             # :1006
```

*Single-convolution path* (`:1030-1040`): `Λ = Σ_τ α_τ mixed[t−τ]` over
τ=1..L (or τ=0..L−1 under `renewal_lag0`); `offset = log((Λ+1)/(anchor+1))`;
`pred ← pred + offset`. The backbone output is read as a log-growth residual.

*Iterated path* `reniter` (`:1008-1029`): for s=1..h,
`I(t+s) = exp(clamp(logR[:,s,:], ±1.5)) · Σ_τ α_τ buf[τ]`, feed `I` back
through `W` into the buffer; final `g = log((I(t+h)+1)/(anchor+1))`,
broadcast to all h slices.

`L = renewal_lag`; auto = `min(14, window) = 14`, capped by
`span = window − 1 = 19` (`:827-849`). `renewal_gamma` (token `renewres`)
scales the offset, init 1.0 (`:858-859`). `gi_fix`/`giunif` freeze α to a
discretised gamma or uniform and set `requires_grad_(False)` (`:881-898`).
Params: `L` (+1 for `renewres`).

**(d) Level cap** — not in the model; applied at inversion (`train.py:53`,
`:202-203`, `:358-359`). See §3.

### 1.9 Total parameter formula

Validated exactly against three checkpoints:

**P(N, S, h) = 7962 + 64·N + 1121·S + 23·h**
(+902 quantile head, +L renewal, +1 temp)

---

## 2. The exp tokens (`--attn_exp`, comma-separated)

Parsed in three places: `SpatialAttentionModule.__init__` (`models.py:196`),
`MSAGATNet_Ablation.__init__` for renewal tokens (`:846`), and
`Trainer.__init__` for optimiser tokens (`train.py:304-308`).

| Token | Parsed at | Effect |
|---|---|---|
| `nodecay` | `train.py:309-315` | Puts `graph_attention.{u,v,adj_scale,log_attn_temp}` in a `weight_decay=0.0` param group |
| `regpre` | `models.py:353-354` | Replaces the post-softmax L1 with `1e-3 · mean\|U@V\|` — fixed λ, gradient-bearing |
| `regent` | `models.py:355-357` | `1e-3 · mean(normalised row entropy of A)`; minimising it forces non-uniform attention |
| `temp` | `models.py:235-236, 333-334` | Adds `log_attn_temp` (init 0) and multiplies the summed logits by `exp(log_attn_temp)` |
| `initN` | `models.py:252-264` | Scales `u,v` by N — applied in `apply_init_scale()`, called *after* `_init_weights()` (`:911-912`) because the global init would otherwise re-Xavier them |
| `lrxN` | `train.py:306-308, 315, 322` | N× learning rate for the same four attention-shaping params |
| `rankN` | `models.py:216-222` | Sets the U@V bottleneck rank (default = `bottleneck_dim` = 8) |
| `adjstd` | `models.py:314-322` | Row-standardises the adjacency prior before scaling, so its within-row spread is O(1) regardless of graph density |
| `scorenorm` | `models.py:325-329` | Row-standardises the *summed* logits before softmax |
| `multgate` | `models.py:300-304` | `S ← S ⊙ σ(U@V)` instead of `S ← S + U@V` |
| `renewal_lag0` | `models.py:846-848, 1030-1031` | Kernel starts at τ=0 instead of τ=1 (the degeneracy ablation) |
| `renewres` | `models.py:858-859` | Free scalar γ on the renewal offset, init 1.0 |
| `reniter` | `models.py:873, 1008-1029` | Iterated renewal — roll the equation forward h steps |
| `giunif` | `models.py:883-887` | Freeze α to uniform over the lags |
| `gifix` | *not a token* | It is the CLI flag `--gi_fix MEAN SD` (`train.py:717-721`), consumed at `models.py:881-898`; appears in the run token only as a filename suffix (`train.py:568-569`) |

**Current kept config: `nodecay,regpre`** — with `--target_space loggrowth
--quantiles` (23 levels). Set in `src/scripts/campaign.py:170-184`, and the
only variant `src/scripts/paper_tables.py:61` reads.
`doc/attention-revival-summary.md` records it as the winner (4/5 proxy cells,
−6.9% validation; gate entropy 0.854, learned-term variance share 0.780 vs
~0.00) with all nine elaborations worse and `multgate` failing the gate.
`program.md:1-7` records the closure: the fix **did not generalise** on the
5-seed test confirmation, and all 45 renewal comparisons lost on accuracy.

---

## 3. Target space, inversion, quantile loss, validation selection

**Target construction** — `DataBasicLoader.growth_targets` (`data.py:266-287`):

```
y = clip(rawdat[idx], 0, None);  a = clip(rawdat[idx − horizon], 0, None)
g = log((y+1)/(a+1));  anchor = a
```

The anchor is exactly the last row of the input window: `_batchify` sets
`end = idx−h+1`, `start = end−P`, so `X[:, −1, :] = dat[idx−h]`
(`data.py:195-213`). Consistent with the renewal decoder's use of
`raw[:, −1, :]` as anchor (`models.py:1036`).

**Supervision** — `train.py:125-129`: the single lead-h growth target is
`.expand(−1, horizon, −1)`, i.e. **all h refinement slices are supervised
with the same lead-h value**. (`--pprm_supervision multistep` uses per-lead
targets but only in level space, `train.py:130-134`.)

**Inversion** — `train.py:190-203`:

1. `ĝ = y_pred[:, −1, :]` (last slice only, `:185`).
2. Clip to `g_bounds = (min_train g − 0.5, max_train g + 0.5)` per node
   (`:353-355`, applied `:198-200`).
3. `ŷ = (anchor+1)·exp(ĝ) − 1`.
4. **Level cap**: `clip(ŷ, 0, 3.0 · data_loader.max)` (`:53`, `:202-203`,
   `:358-359`). `data_loader.max` is the per-node max over the training
   partition only (`data.py:152`), so the cap is train-only.
   `GROWTH_LEVEL_CAP = 3.0` is documented as selected once on validation,
   globally (`train.py:46-53`).

Quantiles are inverted identically (`:207-219`), with the same lo/hi
broadcast and the same level cap.

**Quantile loss** — `pinball_loss` (`models.py:88-97`):
`mean(max(τ·d, (τ−1)·d))` over `[B,N,Q]`, added to the MSE at
`train.py:139-140` and `:173-175`. Levels: the 23-point CDC FluSight /
Forecast Hub set (`train.py:42-44`). Because the median is tied to the point
forecast, the τ=0.5 term is an L1 penalty on the point forecast added to the
MSE.

**Validation selection** — `train.py:430-441`. Criterion is
`MetricsResult.loss` = `Σ_batches(MSE + attn_reg + pinball) /
Σ_batches(B_b·N)`. Consequences:

- Selection is in **training space** (growth-space MSE), not on the inverted
  RMSE that is reported.
- The normaliser divides an already-batch-averaged loss by `n_samples·N`, so
  the absolute scale depends on batch count; monotone within a run,
  meaningless across runs.
- `attn_reg` is inside the selection criterion. Under `regpre` that is
  `1e-3·mean|U@V| ≈ 7e-4`, comparable to the growth-space MSE — so early
  stopping prefers checkpoints with a *small* graph bias.
- Test metrics are computed and printed at every validation improvement
  (`:435-441`) but never used for selection.

Early stopping: patience 100, max 1500 epochs (`:75-79`, `:445-447`). Best
checkpoint reloaded before the final test evaluation (`:449-453`).

---

## 4. Data pipeline

`DataBasicLoader` (`data.py:20-303`).

- **Window** P = 20 (`train.py:492`, `data.py:58`), **horizon** per dataset
  (`train.py:480-487`).
- **Splits**: train=0.6, val=0.2, test=0.2 (`train.py:513`), chronological,
  index-based: `train_end = int(0.6n)`, `val_end = int(0.8n)`
  (`data.py:89-90`); `train_set = range(P+h−1, train_end)`,
  `valid_set = range(train_end, val_end)`, `test_set = range(val_end, n)`
  (`data.py:143-145`). Validation/test *input windows* reach back into the
  preceding partition — targets do not overlap.
- **Normalisation**: min–max, **train-only** (`data.py:141-159`).
  `dat = (rawdat − min)/(max − min + 1e-12)` applied to all rows; val/test
  values may exceed [0,1]. `peak_thold = mean(train_mx, axis=0)` (`:156`).
- **Smoothing**: none at runtime. LTLA and NHS were smoothed offline with a
  trailing causal 7-day mean; Australia is raw; weekly ILI unsmoothed
  (`doc/preprocessing-audit.md:10-21`). Independently confirmed by value
  signatures: LTLA/NHS non-integer, the three weekly sets integer, Australia
  integer with 8 negative entries.
- **Adjacency loading**: `data.py:100-111`, `data/{sim_mat}.txt`.

### Datasets in `data/` (verified with `np.loadtxt`)

| file | T × N | min | max | mean | adjacency | N | density | mean deg | row-std of row-normalised prior |
|---|---|---|---|---|---|---|---|---|---|
| `japan.txt` | 348 × 47 | 0 | 26635 | 655.3 | `japan-adj.txt` | 47 | 0.099 | 4.7 | 0.0685 |
| `region785.txt` | 785 × 10 | 0 | 16526 | 1008.9 | `region-adj.txt` | 10 | 0.420 | 4.2 | 0.1232 |
| `state360.txt` | 360 × 49 | 0 | 9716 | 223.1 | `state-adj-49.txt` | 49 | 0.106 | 5.2 | 0.0635 |
| `australia-covid.txt` | 556 × 8 | −20 | 9987 | 539.4 | `australia-adj.txt` | 8 | 0.469 | 3.8 | 0.1497 |
| `ltla_timeseries.txt` | 839 × 372 | 0 | 4170 | 85.4 | `ltla-adj.txt` (+100/150/200/250) | 372 | 0.310 | 115.3 | **0.0052** |
| `nhs_timeseries.txt` | 895 × 7 | 0 | 1215.4 | 102.8 | `nhs-adj.txt` (+100/150/200/250) | 7 | 0.388 | 2.7 | **0.1970** |
| `spain-covid.txt` | 122 × 52 | — | — | — | `spain-adj.txt` | 52 | — | — | unused; not in `DATASET_CONFIGS` |

All adjacencies are binary, symmetric, with unit diagonal.
`data/geo/{ltla,nhs}_centroids.csv` + `src/scripts/build_adjacency.py`
reproduce the 150 km matrices (NHS exactly, LTLA at Jaccard 0.9993).

The dataset table matches `elsarticle-template-num.tex:1352-1363` exactly.

---

## 5. Inert vs live

### 5.1 Every regulariser and its gradient

| Regulariser | Where | Gradient non-zero? |
|---|---|---|
| Post-softmax L1, `λ·mean\|A\|` (**default**) | `models.py:359-360` → `train.py:138` | **No, provably.** `A ≥ 0` and rows sum to 1, so over `[B,4,N,N]` the mean is exactly `1/N`. The logit gradient is identically 0 |
| its learnable weight `log_attention_reg_weight` | `models.py:199-201` | Non-zero but multiplies a constant. **Not excluded from weight decay** (`train.py:299-301`), and decay on a *log-domain* parameter pushes λ *upward*. Measured: init `log(1e-5) = −11.513`; trained LTLA v1 = **−8.906**, LTLA v2 = **−10.068**. The docstring claim at `models.py:346-347` that "λ decays to kill even the constant" is backwards |
| `regpre`: `1e-3·mean\|U@V\|` | `models.py:354` | **Yes** — subgradient `sign(U@V)` reaches u,v. Fixed λ, cannot self-annihilate |
| `regent`: `1e-3·mean H(A)/log N` | `models.py:355-357` | **Yes** — forces non-uniformity by construction |
| Adam L2 weight decay 5e-4 | `train.py:281`, `:312-326` | **Yes**, on every parameter with a non-`None` grad. Coupled (not AdamW) |
| Gradient clipping, max-norm 1.0 | `train.py:143` | Live |
| Dropout 0.2 | `models.py:130, 211, 421, 528, 62` | Live |
| Low-rank bottlenecks | structural | Not a penalty; the QKV one is a *parameter increase* |
| Early stopping (patience 100) | `train.py:445-447` | Live |

### 5.2 What is measurably dead in trained checkpoints

Loaded from `save_all/`:

| Parameter | LTLA h=7 v1 (level, paper config) | NHS h=7 v1 | LTLA h=7 `nodecay,regpre` |
|---|---|---|---|
| `u` absmax | **1.5e-25** | **2.6e-41** | 2.34 |
| `v` absmax | **3.4e-40** | **8.2e-41** | 1.90 |
| `adj_scale` | 0.00096 → softplus 0.693 | **0.0 exactly** → softplus 0.693 | 3.142 → softplus 3.185 |
| `fusion_weight` | [9.8e-4, 6.2e-8, 4.3e-19, 1.4e-40] → α ≈ **uniform** | **[0.0, 0.0]** → [0.5, 0.5] | [0.450, −0.182, −0.102, −0.179] → [0.38, 0.20, 0.22, 0.20] |
| `log_attention_reg_weight` | −8.906 (drifted **up**) | −8.116 | **−11.513, unchanged** |
| `log_decay` → decay at lead h | −1.594 → γ=0.203, exp(−γh)=0.24 | −1.666 → 0.22 | −0.300 → γ=0.741, exp(−γ·7)=**0.006** |
| `highway_ratio` → σ | 0.266 → 0.566 | 0.142 → 0.535 | 1.035 → 0.738 |

Reading, for the **paper configuration** (`target_space=level`, no exp
tokens — exactly what `elsarticle-template-num.tex` describes):

1. **U@V graph bias is dead** — annihilated by weight decay because, once the
   softmax is flat, the task gradient on a logit bias vanishes and only the
   decay term survives. Confirmed by E3/E7 (row entropy 1.0000, min 0.9998
   over every row/head/sample on LTLA).
2. **Locality-biased adaptive fusion is dead** — `fusion_weight → 0` so
   `softmax → uniform`. Multi-hop mixing still happens, but with *fixed
   uniform* weights over hops. The manuscript's α₀ > α₁ > α₂ > α₃ claim
   (`.tex:1008`) does not survive training.
3. **The adjacency prior is the only surviving logit term with within-row
   spread** — magnitude `softplus(adj_scale) × row-std(Ã)`:
   **1.313 × 0.0052 = 0.0068 on LTLA** vs **1.313 × 0.197 = 0.259 on NHS**.
   Geography is silenced by graph density exactly where the graph is largest.
   This is the direct inverse of the "self-attenuating prior" argument at
   `.tex:643-669`.
4. **Initialisation scale explains the collapse.** Xavier on `u:[4,372,8]`
   gives bound 0.0447, on `v:[4,8,372]` 0.0367; the resulting `U@V` entries
   have sd ≈ **1.6e-3**, three orders below the O(1) spread a selective
   softmax needs.
5. **PPRM's decay branch is annihilated at the scored lead under log-growth.**
   `exp(−γh)` = 0.006 (LTLA `nodecay,regpre` h=7) and 4.6e-4 (NHS h=14). In
   level space it retains ~20–24%. Note the decay term is `x_last·exp(−γd)`
   where `x_last` is a *min–max-normalised level* — under
   `--target_space loggrowth` this is blended, unscaled, into a log-growth
   prediction (`models.py:565-568`).
6. **`log_attention_reg_weight` becomes a frozen dead parameter under
   `regpre`** — it appears in no loss term, so `.grad` is `None`, Adam skips
   it entirely, and it stays bit-exactly at `log(1e-5)`. Confirmed:
   −11.512925 in every `regpre` checkpoint.
7. **`_init_weights` mis-initialises all normalisation gains.**
   `models.py:935-940` catches any 1-D parameter whose name lacks `'bias'` —
   which includes every `LayerNorm.weight` and `BatchNorm.weight` — and
   re-initialises them to `uniform(−1/√d, 1/√d)`. For `LayerNorm(32)` that is
   ±0.177 with random sign, instead of 1.0. Every normalised branch therefore
   starts at ~0.1× scale and randomly signed. This is a bug, not a design
   choice (the `_PRESERVE_PARAMS` list at `:915-925` shows the intent was to
   guard the *scalar* inits).

**What is genuinely live in the paper config**: TFEM, the rank-8 feature
projection, the QKV/value/out projections (i.e. *uniform-mean-pooled* spatial
aggregation with a residual), the per-hop MSSFM transforms (uniformly
averaged), PPRM's projection + gate + decay, and the highway AR blend. In
other words: an MLP over one 3-tap temporal filter, plus fixed uniform
spatial smoothing, plus a 4-tap AR.
`doc/attention-revival-summary.md` reaches the same conclusion — "the module
is a global mean-pooling layer wearing an attention costume" — and shows the
aggregation is nonetheless worth +27.1% over `no_agam` in v2 space.

---

## 6. Hyperparameters: code vs manuscript

| Item | `train.py` / `models.py` | Manuscript | Verdict |
|---|---|---|---|
| lr, batch, epochs, patience, wd | 1e-3, 32, 1500, 100, 5e-4 (`train.py:490`) | identical (`.tex:1408-1415`) | match |
| window, splits | 20, 0.6/0.2/0.2 (`train.py:513-514`) | identical (`.tex:1417`) | match |
| `d_feat`, kernel, `d_bottle`, `d_hidden`, heads | 16, 3, 8, 32, 4 (`train.py:491-492`) | identical (`.tex:1427-1436`) | match |
| `S_max`, fusion init | 4, `exp(−0.5k)` (`models.py:413,426`) | identical (`.tex:1438-1440`) | match |
| PPRM γ₀, dropout | −2.3, 0.2 (`models.py:533`) | identical (`.tex:1443-1445`) | match |
| Highway `w_h`, gate λ | `min(4,w)`, init 0.5 pre-sigmoid | "λ initialised to 0.5" (`.tex:1448-1449`) | ambiguous — effective blend at init is 0.622 |
| **Attention regularisation weight** | `1e-5` learnable in log domain, **term added to the loss** | Table lists it as a hyperparameter (`.tex:847`) but the text states sparsity is promoted "without requiring an explicit sparsity penalty term in the loss function" (`.tex:831`) | **Mismatch.** The code has an explicit penalty; it is just inert |
| **Bias rank `d_bias`** | 8, tied to `bottleneck_dim` (`models.py:217`) | `d_bias ≪ N`, never given a value (`.tex:641`) | not reported |
| **Loss** | `MSE + attn_reg (+ pinball)` (`train.py:138-140`) | `MSE` only (`.tex:1410-1413`, Alg. 1 `.tex:1477`) | mismatch |
| **Target space** | paper runs use `level`; kept config uses `loggrowth` | level only | manuscript describes v1 |
| **Quantile head / probabilistic output** | present, 23 CDC levels | absent | manuscript describes v1 |
| **Renewal decoder** | present (`models.py:825-907`) | absent | manuscript describes v1 |
| **Level cap 3× train max** | applied at inversion (`train.py:53`) | absent | not documented |
| **Adaptive hop depth S** | S=2 for N=10 (`models.py:413`) | `.tex:1000` correctly states S=2 for US-Regions | fixed (earlier contradiction recorded in the ledger) |
| **QKV bottleneck saves parameters** | 3192 vs 3168 | `.tex:606` claims a significant reduction | **false at these dimensions** |
| **Notation** | heads = 4 | `U ∈ R^{h×N×d_bias}` (`.tex:641`) uses `h` for heads, colliding with horizon `h` | notation collision |
| **"Learnable U,V alone are sufficient"** | u,v are 1e-25…1e-41 in every level-space checkpoint | `.tex:1604` concludes U,V alone suffice | **contradicted.** Both arms of that comparison have a dead U@V; it compares uniform mean pooling ± an adjacency logit |
| **"O(N) linear complexity"** | dense `[B,4,N,N]` softmax and dense `Â^k X`; N² buffers | `README.md:7` | **false; the model is O(N²d)** |

---

## 7. Parameter counts

**P(N, S, h) = 7962 + 64N + 1121S + 23h**, plus 902 (quantile head), L (renewal).

Validated exactly:

- `save_all/…ltla_timeseries…h-7.none.seed-42.with_adj.pt` → 36,415
- `save_all/…nhs_timeseries…h-7.none.seed-42.with_adj.pt` → 10,813
- `save_attn/…nhs_timeseries…h-14…exp-nodecay-regpre.pt` → 11,876 (= 10,974 + 902)

### LTLA (N=372, S=4)

| config | h=3 | h=7 | h=14 |
|---|---|---|---|
| paper (level) | 36,323 | **36,415** | 36,576 |
| + quantile head (kept config) | 37,225 | **37,317** | 37,478 |
| + renewal (L=7) | — | 37,324 | — |

EAGAM alone is **27,618** = 76% of the model, of which u,v = 23,808 (**65% of
all parameters**) — and those are the ones weight decay annihilates in the
paper configuration. Non-parameter buffers: `adj_prior` + 4 hop matrices =
691,920 floats ≈ 2.8 MB.

### NHS (N=7, S=2)

| config | h=3 | h=7 | h=14 |
|---|---|---|---|
| paper (level) | 10,721 | **10,813** | 10,974 |
| + quantile head | 11,623 | 11,715 | **11,876** |
| + renewal (L=7, `reniter`) | — | — | 11,883 |

Here u,v = 448 (4%). Others: Japan (47, S=4) h=5 → 15,569; US-States (49,
S=4) h=5 → 15,697; Australia (8, S=2) h=7 → 10,877; US-Regions (10, S=2)
h=5 → 10,935.

---

## 8. Honest architectural assessment

### Standard, with clear precedent

| Component | Precedent |
|---|---|
| Additive learnable bias on pre-softmax attention logits | Graphormer spatial-encoding bias (Ying et al., NeurIPS 2021); T5 relative-position bias (Raffel et al. 2020). MSAGAT's version is per-head and node-pair-indexed — the same object Graphormer calls `b_{φ(i,j)}` |
| Low-rank `U@V` learned graph structure | Graph WaveNet's adaptive adjacency (Wu et al., IJCAI 2019); MTGNN (KDD 2020); AGCRN (NeurIPS 2020). The manuscript cites these at `.tex:252` and differentiates on "combining learnt structure with an optional prior" — which is Graphormer's move applied to Graph WaveNet's parameterisation |
| Powers of the normalised adjacency with learned per-hop mixing | DCRNN diffusion convolution (ICLR 2018); MixHop (ICML 2019) is `concat_k Â^k X W_k`; APPNP/SGC. MSSFM is MixHop with a softmax-weighted sum instead of concatenation |
| Depthwise separable temporal conv | Xception (Chollet, CVPR 2017), cited at `.tex:469` |
| Low-rank QKV bottleneck | Bhojanapalli et al., "Low-Rank Bottleneck in Multi-head Attention" (ICML 2020); Linformer (2020) |
| Sigmoid-gated linear AR skip ("highway") | LSTNet's autoregressive component (SIGIR 2018) essentially verbatim; Cola-GNN carries the same. Both are baselines in this paper |
| Gated blend of a learned forecast with a decaying-persistence extrapolation | ES-RNN (Smyl, IJF 2020); N-BEATS trend basis (ICLR 2020); damped-trend exponential smoothing (Gardner & McKenzie 1985) — `x_last·exp(−γd)` *is* a damped trend with a learned damping |
| Monotone quantile head via cumulative softplus offsets | Spline-quantile-function RNNs (AISTATS 2019); MQ-R(C)NN (2017) |
| Pinball loss over 23 quantiles | CDC FluSight / COVID-19 Forecast Hub convention (Bracher 2021; Cramer 2022), cited at `train.py:40-41` |
| Log-growth target | Standard log-ratio/growth-rate targeting. The repo's own evidence (E2) says it is the largest measured effect and transfers across architectures — a good result, but a known transformation |
| Renewal equation | Fraser (2007); Cori et al. (2013, EpiEstim) |
| Haversine-threshold geographic graph | STAN (Gao et al. 2021), cited at `.tex:1391` |

### Not standard, but also not a contribution

- **"Self-attenuating additive adjacency prior"** (`.tex:643-669`). The
  mechanism is softmax shift-invariance — textbook. And the empirical
  direction is the opposite of the claim: attenuation is a *failure*
  (row-std 0.0052 on LTLA vs 0.197 on NHS), and it happens on the large
  graphs where structure ought to matter most.
- **`regpre`** (pre-softmax L1 on U@V). L1 on learned graph-structure logits
  is standard in graph structure learning (NRI, GTS, LDS). Here it is a bug
  fix for an inert penalty.
- **`adjstd`** — density-invariant prior standardisation. Correct, useful,
  and two lines.
- **The level cap.** A necessary decoding guard for a multiplicative target
  space, not a modelling idea.

### The one genuinely-not-in-prior-work candidate

**The iterated, spatially-coupled renewal decoder with a learned
generation-interval kernel on a GNN backbone** (`models.py:1008-1029`,
`:1048-1072`): the backbone predicts `log R_i(t+s)`, a softmax-over-lags α is
a proper delay distribution, a row-stochastic learned `W` couples regions,
and the equation is rolled forward h steps with prediction feedback. No
direct precedent for exactly this combination — closest relatives are
MepoGNN, STAN, and discrete-time Hawkes processes with learned triggering
kernels.

Three caveats the repository itself records and this audit confirmed in code:

1. **It loses.** Single-convolution 30 comparisons / 0 wins; iterated 45
   comparisons / 0 wins, and the *correct* formulation is worse than the
   incorrect one (+20.6% vs +12.5%).
2. **The generation-interval reading was withdrawn** for the
   single-convolution form. Only the `reniter` path restores a genuine delay
   kernel — and the measured NHS h=14 kernel is
   `[0.169, 0.212, 0.248, 0.130, 0.095, 0.073, 0.074]`, plausible in shape
   (mode at lag 3) but achieved at a loss in accuracy.
3. `--renewal` is off in every configuration that feeds the AIIM manuscript.

### Bottom line

The AIIM manuscript describes a recombination of Graphormer's attention bias,
Graph WaveNet's low-rank adaptive adjacency, MixHop/DCRNN multi-hop
diffusion, Xception separable convolutions, and LSTNet's AR highway — each
individually well-precedented, none cited as the source of the specific
mechanism (Graphormer, MixHop and Bhojanapalli et al. do not appear in the
Methods citations at all). More seriously, in the exact configuration the
manuscript describes, **the two components the novelty claims rest on — the
learnable U@V graph bias and the locality-biased adaptive hop fusion — are
numerically zero in every trained checkpoint loaded**, so the reported
results are those of a model whose spatial pathway reduces to a fixed uniform
mean pool plus a fixed uniform hop average, plus a geographic logit that is
itself density-silenced on the largest dataset.
