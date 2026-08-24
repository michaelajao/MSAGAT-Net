# 05 — Provenance audit: how results, tables and figures are produced and saved

Read 24 Aug 2026: every writer in `src/` (grep for `savefig`, `to_csv`,
`savez`, `torch.save`, `open(...,'w')`), plus the artefacts on disk.

---

## 0. The verdict in one paragraph

There is **no run manifest, no git hash, no hyperparameter record and no code
version stored with any result**. The only identity a run carries is the
`log_token` string built at `src/train.py:551-571`, and that token omits at
least four things that change the numbers (`GROWTH_LEVEL_CAP`, the quantile
*count*, `--eval_only` vs trained-fresh, and all of `TRAIN_DEFAULTS`).
`report/`, `doc/`, `save_all/`, `save_attn/`, `save_renewal/`, `*.pt` and
`*.log` are **all gitignored** (`.gitignore:41-64`); `git ls-files report`
returns 0 files; the last commit `d142c18` is dated 2026-08-12 — *before* the
attention-revival and renewal campaigns — and `src/train.py`, `src/models.py`,
`src/scripts/dm_test.py`, `src/scripts/campaign.py` are all dirty, with 7 of
the paper scripts untracked. **The code that produced every number in both
papers exists only in the working tree.**

---

## 1. Writer table

### Run identity: the canonical token

Built once, at `src/train.py:551-571`:

```
{model_name}.{dataset}.w-{window}.h-{horizon}.{ablation}.seed-{seed}.{adj_tag}{sim_tag}{variant_tag}
```

| component | source | rule |
|---|---|---|
| `model_name` | `train.py:546` | always literal `MSAGAT-Net` |
| `w-{window}` | `:570` | always 20 |
| `adj_tag` | `:551` | `with_adj` / `no_adj` |
| `sim_tag` | `:552` | `.{sim_mat}` **only if `--sim_mat` explicitly passed** |
| `.pprm-{sup}` | `:554-555` | only if ≠ `repeat` |
| `.sgate` | `:556-557` | |
| `.{target_space}` | `:558-559` | only if ≠ `level` → `.loggrowth` |
| `.quant` | `:560-561` | **presence only — no level count** |
| `.attnfix` | `:562-563` | |
| `.exp-a-b-c` | `:564-565` | `attn_exp` with `,`→`-` |
| `.renewal{lag}` | `:566-567` | `renewal_lag or ''`, so lag 0 → bare `.renewal` |
| `.gifix{m:g}-{sd:g}` | `:568-569` | |

**The same grammar is re-implemented eight times** — see
[the inventory](#see-also) — and `campaign.py:46-63` (`token_for`) **omits the
`renewal` and `gi_fix` suffixes**, a known divergence.

### Artefact writers

| Artefact | Writer | Path template | Identity carried | Write mode | Dedup key |
|---|---|---|---|---|---|
| **Prediction `.npz`** | `train.py:613-614`, payload `:591-612` | `report/predictions/{dataset}/{log_token}.npz` | filename = full token; payload also stores `model, dataset, horizon, window, seed, ablation, use_adj, sim_mat, pprm_supervision, spatial_gate, target_space, protocol='lead_h'`, `quantile_levels` if quantiles, and `y_true_val/y_pred_val/y_pred_q_val` | **overwrite, silent** | filename |
| **Checkpoint `.pt`** | `train.py:461-464` | `{save_dir}/{log_token}.pt`; `save_dir` ∈ `save_all` / `save_attn` / `save_renewal` | filename only — a bare `state_dict`, no metadata inside | **overwrite; rewritten on every val improvement** | filename |
| **`best_model.pt`** | `train.py:465-466` | `{save_dir}/best_model.pt` | **none** | overwritten by *every* run and *every* improving epoch | none — worthless, and a footgun |
| **Metrics CSV row** | `utils.py:526-606`, called `train.py:618-620` | `report/results/{dataset}/all_results.csv` | `model` (= `MSAGAT-Net` + `variant_tag`, `train.py:616`), `dataset, window, horizon, ablation, seed, use_adj, sim_mat, timestamp` + metrics | append → dedup → **full rewrite** (`:600-606`) | `(model, dataset, window, horizon, ablation, seed, use_adj, sim_mat)` (`:589-598`) |
| **Metrics TXT line** | `utils.py:609-629` | `report/results/{dataset}/all_results.txt` | timestamp + 4 metrics | **pure append, never deduped** → diverges from the CSV | none |
| Per-run metrics CSV | — | `report/results/{dataset}/final_metrics_{log_token}.csv` | **never written.** `utils.py:548` uses only `os.path.dirname(save_path)`. `train.py:623` prints "Results saved to {results_csv}" — a false message | — | — |
| **Attention `.npy`** | `extract_attention.py:106` | `report/attention/{ckpt_token}.npy` | filename | skip-if-exists unless `--force` | filename |
| **Campaign log** | `campaign.py:250-252` | `report/logs/campaign_{chunk}.log` | `{stamp} {token} {ok\|FAIL rc=} {secs}` | append | none. **Per-run stdout is `DEVNULL`** (`:245-246`) — tracebacks discarded |
| **Baseline campaign log** | `baseline_campaign.py:150-154` | `report/logs/baseline_campaign.log` | `{stamp} {tag}.{ds}.h-{h}.seed-{s} status secs` | append | none |
| **Per-run baseline log** | `baseline_campaign.py:137-141` | `report/logs/runs/{tag}.{ds}.h-{h}.seed-{s}.log` | full stdout+stderr | `'w'` — **overwrite**; only 22 files survive | filename |
| **`attn_revival_runs.csv`** | `attn_revival.py:271` → `append_row` `:158-194` | `report/results/attn_revival_runs.csv` | `stamp,kind,exp,dataset,horizon,val_rmse` + gate cols | append; header widened by rewrite if new cols | **none** — 15 of 30 rows are duplicates |
| **`attn_revival_ledger.csv`** | `attn_revival.py:241-246, 298` | same mechanism | `stamp,kind,exp,mean_val_rmse,5 cell cols,ent_*,uv_share,alpha_*,gate_pass,note` | append | none |
| ↑ **rebuilt by hand** | `rebuild_ledger.py:27-173` | same path, `'w'` | 26 rows are **Python literals** transcribed from session-monitor events | **destructive overwrite** | none |
| **`conformal_metrics.csv`** | `conformal.py:189-190` | `report/results/conformal_metrics.csv` | `dataset,horizon,seed,method` + WIS/coverage | **full overwrite of the whole file** | none |
| **`prob_metrics.csv`** | `prob_eval.py:113-114` | `report/results/prob_metrics.csv` | `model,dataset,horizon,seed,ablation,variant,split,n_test` + WIS/cov | **full overwrite** | none |
| **DM CSV** | `dm_test.py:250-252` | `report/results/dm_tests_{variant}_{arms}.csv` | `dataset,horizon,baseline,baseline_arm,msagat_seed,baseline_seed,…` | full overwrite of that variant file | filename encodes variant+arms |
| **`epiestim_validation.csv`** | `epiestim_check.py:206-211` | `report/results/epiestim_validation.csv` | `dataset,horizon` only (5 rows) | **full overwrite** | one row per (ds,h) |
| **`aggregated_multiseed_results.csv`** | `evaluate.py:390-392` | `report/results/aggregated_multiseed_results.csv` | `dataset,horizon,ablation,n_seeds,seeds` + mean/std | full overwrite | pooling key `:341-344` |
| **Adjacency `.txt`** | `build_adjacency.py:74-75` | `data/{adj_name}-{thr}.txt` | threshold in filename | overwrite | filename |
| **Paper A tables** | `paper_tables.py:25-29` | `doc/plos-renewal/tables/*.tex` | none inside | overwrite | — |
| **Paper A figures** | `paper_figures.py:39-43` | `doc/plos-renewal/figs/F{2,3,4,5}_*.{pdf,png}` | none inside | overwrite | — |
| **Paper figures (Elsevier era)** | `evaluate.py:107-111` | `report/figures/paper/{name}.png` | none | overwrite | — |
| **Diagnostic figures** | `evaluate.py:275-276, 313-316`; helpers `utils.py:160, 356, 459` | `report/figures/{dataset}/{matrices,predictions,predictions_summary}_{dataset}_h{h}.png` | dataset+horizon **only — no seed, no ablation, no variant** | overwrite | — |
| **Loss curves** | `utils.py:470-512` | — | **dead code.** `plot_loss_curves` is imported at `train.py:36` and **never called**. Every `loss_curve_*.png` on disk is orphaned | — | — |
| **TensorBoard** | `train.py:376-378, 428` | `tensorboard/{log_token}/` | token | append (626 dirs) | token |

---

## 2. Confirmed defects

### (A) `dm_tests_v2_best.csv` does not cover the attention-fixed arm

`dm_test.py:49` — `VARIANTS = {'v1': 'with_adj', 'v2': 'with_adj.loggrowth.quant'}`,
and `msagat_re` (`:57-60`) anchors the regex with `$`, so
`…quant.exp-nodecay-regpre.npz` **cannot match**. Verified numerically:

| arm | australia-covid h3 seed45 test RMSE |
|---|---|
| `with_adj.loggrowth.quant` | **107.7925** |
| `…quant.exp-nodecay-regpre` | 97.3294 |

`dm_tests_v2_best.csv` row 2 carries `msagat_rmse = 107.79253888701206` → it
is the **frozen v2 arm**. That is correct for Paper B (frozen v2 *is* the
paper's model), but **no DM or paired test exists anywhere for the
`exp-nodecay-regpre` arm**, which is what the horizon-threshold claim (E11)
rests on. Paper B needs one.

### (B) `prob_metrics.csv` and `conformal_metrics.csv` are stale by one campaign

`prob_metrics.csv` (14 Aug) contains exactly one `variant`:
`with_adj.loggrowth.quant`, 105 rows. `prob_eval.py`'s `TOKEN_RE` (`:25-27`)
*would* match the exp arm — it simply has not been re-run. `conformal.py`'s
`TOKEN_RE` (`:40-42`) is `$`-anchored and so **structurally excludes** it, and
`conformal_metrics.csv` has **no `variant` or `model` column at all**, so
nothing on disk records which arm it describes.

### (C) Partial runs silently destroy the whole file

Both `conformal.py:190` and `prob_eval.py:114` write with `df.to_csv(OUT_CSV)`
and no merge. Running `python -m src.scripts.conformal --dataset
nhs_timeseries --horizon 3` replaces all 450 rows with ~4. Same for
`prob_eval --split val`.

### (D) `--eval_only` re-scoring already overwrote originals, silently

`train.py:391-401` returns early with re-evaluated metrics, then
`run_single_experiment` falls through to `:588-620` and rewrites the **same
npz** and the **same CSV dedup key** with a fresh timestamp.
`report/logs/rescore_v2.log` is the receipt: "105 v2 checkpoints to
re-score", 14 Aug 03:33, and its line 10 (`australia-covid h=3 seed=45 …
RMSE 107.7925`) matches the current npz byte-for-byte. **The original
pre-rescore predictions no longer exist and nothing on disk marks a row as a
rescore.**

### (E) The level cap is invisible

`GROWTH_LEVEL_CAP = 3.0` (`train.py:53`), applied `:358-359`, used `:203`,
`:215-216`. **Not** in the token, **not** in the npz payload, **not** a CSV
column, **not** a CLI flag. Changing it and re-running `--eval_only` rewrites
every `loggrowth` npz and CSV row with identical names and keys. Same for
`g_bounds` ±0.5 (`:354-355`).

### (F) Quantile count is not in the token

`train.py:560-561` writes only `.quant`. `extract_attention.py:84-89`
documents that dev runs used 7 levels and the v2 campaign 23, and has to
*infer* the count from `quantile_head.proj.3.bias`. A 7-level and a 23-level
run with the same seed/dataset/horizon **collide on filename and CSV dedup
key**. The npz stores `quantile_levels`, so the npz is recoverable; the CSV
row is not.

### (G) `attn_revival_runs.csv` has no dedup and 50% duplicates

15/30 rows repeat `(kind, exp, dataset, horizon)` with identical `val_rmse`
and different `stamp` — `attn_revival.py:258-262` skips the run when the npz
exists but still appends a row. Anything grouping on `exp` double-counts.

### (H) The ledger is hand-typed and not machine-linkable

`rebuild_ledger.py` reconstructs 26 rows from *session-monitor transcripts*
(docstring `:1-7`) as Python literals. Its `exp` column mixes tokens with
prose — `'nodecay,regpre + renewal lag14 tau>=1'` — which is **not** an
`attn_exp` string any script accepts, so no ledger row maps mechanically to
an npz or `.pt`. Worse, `'nodecay,regpre,temp'` appears **twice** (exp-2 and
exp-2R) and `'nodecay,regpre,init10'` twice (exp-3, exp-3R) with different
numbers, disambiguated only by free-text `[INVALIDATED…]` notes. Since both
members of each pair write the *same* token, **exp-2R's checkpoint and npz
overwrote exp-2's**; the invalidated runs are unreproducible.

### (I) `paper_tables.py` reads the ledger by fixed column index

`:172` — `float(row[4 + i])` against `CELL_ORDER` (`:154-155`), with rows
located by scanning for a note starting `R2:`/`I1 ` (`:158-163`). Any reorder
of `H` in `rebuild_ledger.py:16-19`, or any prose edit to a note, silently
produces wrong numbers or exits.

### (J) Hard-coded numbers inside the "never hand-typed" table generator

- `paper_tables.py:131-133` — `BARV = {...}`, the exp-1 baseline validation
  RMSEs, typed as literals. Every `+12.5%` / `+35.0%` in
  `accuracy_sweep.tex` is computed against typed constants, not artefacts.
- `paper_tables.py:216` — `'Implied $R$ of the renewal decoder & 0.942'`
  typed into `naive_baselines.tex`, while `epiestim_validation.csv` holds
  `0.9416269…`. Right today, unlinked forever.

### (K) `all_results.csv` schema drift — confirmed

- `MAPE` is a live column but `MetricsResult.to_dict()` (`train.py:104-111`)
  no longer emits it (commit `51cf375` "Remove MAPE metric"). Non-null counts:
  japan 124/269, nhs 48/258, australia 74/143 — **the column silently
  partitions old runs from new**.
- `sim_mat` is missing entirely from `report/results/spain-covid/all_results.csv`;
  back-filled in memory (`utils.py:583-584`), never on disk.
- `sim_mat` is `'default'` in **every** row of every file → the
  `chunk_sensitivity` threshold runs were **never executed**. No
  adjacency-threshold sensitivity result exists.
- `evaluate.py:78-87` filters on `model == 'MSAGAT-Net'` — the bare string —
  so it can only ever see the level-space v1 arm, never `.loggrowth.quant…`.

### (L) The seed literal 42 is a magic default

`utils.py:558` (`seed or 42`), `:577`, `:594`. A run genuinely seeded `0`
would be recorded as 42 (`0 or 42 == 42`) and collide with the real seed-42
row.

### (M) `n_test` is not recorded in `all_results.csv`

It *is* in `prob_metrics.csv` and `dm_tests*.csv` (`dm_test.py:234`), which is
how a cross-check is possible; the core metrics table cannot be checked
against a protocol change.

---

## 3. Missing-provenance matrix

| Thing that changes results | In token? | In npz? | In CSV? | Anywhere? |
|---|---|---|---|---|
| dataset / window / horizon / ablation / seed / use_adj | yes | yes | yes | |
| `sim_mat` (non-default) | yes | yes | yes | |
| `target_space`, `pprm`, `sgate`, `attnfix`, `attn_exp`, `renewal`, `gi_fix` | yes | partial | via `model` col | |
| quantile **count** | **no** | yes | **no** | npz only |
| `GROWTH_LEVEL_CAP` | **no** | **no** | **no** | **nowhere** |
| `g_bounds` ±0.5 margin | **no** | **no** | **no** | **nowhere** |
| `TRAIN_DEFAULTS` (epochs/lr/wd/batch/dropout/hidden/heads/scales) | **no** | **no** | **no** | **nowhere** |
| `attention_regularization_weight`, `adj_weight`, `highway_window` | **no** | **no** | **no** | **nowhere** |
| `--eval_only` (rescore vs fresh train) | **no** | **no** | **no** | only `rescore_v2.log`, gitignored |
| best epoch / epochs run / early-stop | **no** | **no** | **no** | TensorBoard only |
| conformal `LAM/GAMMA/WINDOW/TOP_K` | — | — | **no** | **nowhere** |
| DM `alpha`, `--arms`, `--variant` | — | — | filename only | |
| **git commit / code version** | **no** | **no** | **no** | **nowhere; `report/` is gitignored and HEAD predates the work** |
| wall-clock timestamp | — | **no** | yes | CSV + campaign logs |

---

## 4. Figures

### Figure-generating scripts

| Script | Output | Reads | Reproducible? |
|---|---|---|---|
| `evaluate.py:116-142` | `report/figures/paper/fig1_{rmse,pcc}_vs_horizon.png` | `all_results.csv` via `load_metrics` (seed 42, `model=='MSAGAT-Net'`) | yes, but pinned to the **v1 level-space arm only** |
| `evaluate.py:145-175` | `fig2_ablation_{ds}_h{h}.png` | ditto | **partially orphaned** — the loop is hard-coded to japan, yet 19 files exist for australia, ltla, nhs, region785, **spain-covid**, **state360**. Those cannot be regenerated. Dated Feb 2026 |
| `evaluate.py:178-223` | `fig6_component_impact_{ds}.png` | ditto | same problem — 7 files exist incl. datasets outside the loop |
| `evaluate.py:254-277` → `utils.py:105-164` | `report/figures/{ds}/matrices_{ds}_h{h}.png` | checkpoint via `_model_path` (**v1 token, no variant tag**) | v1 only. `_extract_attention` (`utils.py:167-210`) falls back to **zeros** with only a `logger.warning`, and is called with `logger=None` (`evaluate.py:276`) — **a silent all-zero "learned attention" panel is possible** |
| `evaluate.py:280-317` → `utils.py:233-360, 363-463` | `predictions_{ds}_h{h}.png`, `predictions_summary_{ds}_h{h}.png` | checkpoint + loader; **re-runs the model rather than reading the npz** | inconsistent with the npz pipeline — does its own forward and slicing with **no growth inversion**, so these plots are in a different target space from the reported metrics |
| `utils.py:470-515` | `loss_curve_*.png` | — | **dead code, never called.** All ~120 files reference `MSTAGAT-Net`/`EpiDelay-Net`/`seed-5` — a model name and seed that appear nowhere in current code |
| `paper_figures.py` | `doc/plos-renewal/figs/F{2,3,4,5}_*` | see below | yes |

### Paper A figures on disk

| File | Shows | Reads |
|---|---|---|
| `F2_gi_kernels` | learned α_τ per seed + mean, 3 horizons | `renewal_paper.kernel()` → `save_renewal/{token}.pt['log_alpha']`, falling back to `save_attn/` |
| `F3_r_validation` | model implied R vs EpiEstim, NHS h=3 | `save_attn/*reniter*.pt` picked by a **fragile ad-hoc filter** (`paper_figures.py:79-89`, an `and`/`or` precedence tangle) + `data/nhs_timeseries.txt` |
| `F4_arms_test` | test RMSE, 4 arms × 3 horizons | `report/predictions/nhs_timeseries/*.npz` |
| `F5_horizon_threshold` | (a) Δ test-RMSE of the attention fix; (b) learned γ vs horizon | npz regex + `save_attn/*renewres*.pt['renewal_gamma']` |

**No F1** — Figure 1 of `main.tex` is the hand-drawn TikZ architecture diagram.

**`main.tex` prose carries numbers that no artefact holds**:
`3.23±0.06 / 4.52±0.34 / 3.62±0.37` and "14 of the 15 models" (`:266-268`,
duplicated from `gi_recovery.tex`, so a table regen will not update the
prose); "all seven NHS regions exceed r=0.5 … median r=0.90 … 0.71"
(`:370-372`, produced only by `epiestim_check.py --per-region`, which
**prints to stdout and writes nothing**); "+12.6%, −2.6%, −6.1%" and
"γ≈0.65–0.70 … recovers to 0.20" (`:465-473`, computed into local dicts in
`paper_figures.py:176-201` and rendered into the PNG only — never persisted).

### Paper B figures

**6 hand-drawn TikZ figures** (`.tex:302, 520, 673, 852, 1026, 1211`), drawn
inline.

**16 `\includegraphics`**, all from `msagat_figs/`:

- `predictions_summary_*` (6) — same basenames as `report/figures/{ds}/`, so
  `evaluate.py:315` is the generator, but the copies are dated Feb 2026 and
  differ from the `report/` copies. **Hand-copied, not built.**
- `matrices_MSAGAT-Net.japan.w-20.h-7.{none,no_pprm,no_agam}.png` and the LTLA
  one (4) — **filenames no current code can produce.** `evaluate.py:275`
  emits `matrices_{dataset}_h{horizon}.png`; these were **renamed by hand**,
  and the ablation variants are not generated at all (`fig_diagnostic_matrices`
  only ever passes `'none'`).
- `fig6_component_impact_*` (3) — copies of `report/figures/paper/`.

**All 7 Paper B tables are hand-typed.** `grep -c '\input{'` → **0**;
`grep -c 'begin{tabular}'` → 7. There is no script anywhere that emits an
Elsevier table.

---

## 5. Recommended provenance scheme

No change to any evaluation or DM statistic. Everything is additive.

1. **One manifest per run** — `report/manifests/{dataset}/{token}.json`
   written next to the npz: git commit + dirty flag, argv, full config,
   constants (level cap, g_bounds margin, quantile levels, `TRAIN_DEFAULTS`),
   mode train/eval_only, best_epoch / epochs_run / n_params, n_train/val/test,
   adjacency file + sha256, artefact paths + sha256, final metrics.
   `best_epoch` and `epochs_run` are already tracked on the `Trainer`
   (`train.py:384-386`) — just return them.
2. **Extend the token** so filenames stop colliding: append `.q{n}` when the
   quantile count ≠ 23 and `.cap{c}` when the level cap ≠ 3.0. Both are
   no-ops for existing runs, so **no file on disk is renamed**.
3. **Three extra columns in `all_results.csv`** — `git_commit`, `run_token`,
   `mode`. Keep the dedup key as-is, but log a warning when a row is replaced
   by one with a different commit. That single warning would have caught
   defect (D).
4. **A canonical index** — `report/results/runs_index.csv`, derived (never
   authored) by walking the manifests and cross-checking each against the npz
   and the CSV row, with a `consistency` column. Primary key `run_token`.
   Every downstream table and figure joins against it.
5. **`make_all.py`** — `--check` verifies `config/expected_runs.yaml` against
   the index and fails with the missing tokens; full mode regenerates every
   table and figure and scans emitted `.tex` for `nan|inf|0.00 \pm 0.00`.
6. **Make writers non-destructive** — merge-on-write in `conformal.py` and
   `prob_eval.py` (the pattern `utils.py:573-606` already uses); delete the
   `best_model.pt` write; stamp the per-run baseline log filename.
7. **Freeze the reconstructed ledger** as
   `report/results/frozen/attn_revival_ledger.v1.csv` + a `PROVENANCE.md`
   stating that exp-2/exp-3 were overwritten by their reruns and are not
   reproducible; add a `run_token` column; make `paper_tables.py` read by
   column name.
8. **De-hardcode** `BARV` and `0.942`; persist `horizon_threshold.csv` and
   `epiestim_per_region.csv` so the prose numbers have sources.
9. **Track the small artefacts** — replace the blanket `report/` and `doc/`
   ignores with an allow-list for `report/results/**/*.csv`,
   `report/manifests/**/*.json`, `paper/**/*.tex|*.bib`, `docs/**`. About
   1 MB of CSV + JSON. Checkpoints and npz stay out; their SHA-256s live in
   the tracked manifests, so a claim can be verified against a checkpoint even
   when the checkpoint is not in the repository.

---

## See also

The repository inventory (module map, the eight token builders, the five npz
loaders, dead and stale modules, doc overlap, hygiene) informed the
reorganisation plan and is summarised in the approved plan file rather than
duplicated here.
