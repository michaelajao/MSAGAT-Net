# doc/ — canonical documents

## Papers

| Path | Paper | Venue | Status |
|---|---|---|---|
| `plos-renewal/` | **Paper A** — a neural forecaster that learns its own generation interval | **JAMIA** (decided 25 Aug 2026) | LaTeX drafted and built in PLOS format; needs the structural conversion to JAMIA (Methods before Results, structured abstract, no Author Summary, ~4000-word limit). Directory keeps its `plos-renewal` name until the move to `paper/jamia/` |
| `elservier/` | **Paper B** — the evaluation paper: corrected protocol, target-space study, calibration, attention failure mode | **Computers in Biology and Medicine** (fresh submission; the Elsevier transfer offer is declined) | rewrite pending |
| `paper-c-design.md` | **Paper C** — identifiability-constrained likelihood decomposition + cross-outbreak transfer | Neurocomputing / Knowledge-Based Systems | design only; not started until Paper B's protocol exists |

All three venues are covered in full by the Coventry read-and-publish
agreements, so no article processing charge is payable.

## Audit of 24–25 August 2026

`audit-2026-08-24/` — the evidence base for the three-paper programme. Read
this before making any claim about what the model does or what the field has
already published.

| File | Contains |
|---|---|
| `audit-2026-08-24/01-docs-and-results-digest.md` | Findings E1–E12, the dead claims, the current DM grid, calibration numbers, 13 contradictions between documents |
| `audit-2026-08-24/02-architecture-audit.md` | Module-by-module forward pass, parameter formula, what is inert vs live, code-vs-manuscript mismatches |
| `audit-2026-08-24/03-literature-baselines-and-followons.md` | Cola-GNN, EpiGNN, MepoGNN, STAN, CNNRNN-Res, DCRNN, LSTNet + the 2023–2026 follow-ons and benchmark papers |
| `audit-2026-08-24/04-user-papers-folder.md` | The eight PDFs in `GNN forecasting/` plus Fritz et al. 2022 |
| `audit-2026-08-24/05-provenance-audit.md` | How every artefact is written; 13 confirmed defects; the figure inventory |
| `audit-2026-08-24/06-data-provenance.md` | Where the NHS, LTLA and Australia series actually came from |

## Research record (current, canonical)

| File | Role |
|---|---|
| `MSAGAT-RESEARCH-LEDGER.md` | Single source of truth: findings E1–E14, decisions, outcomes of both autoresearch programmes |
| `attention-revival-summary.md` | Attention campaign: the two-line fix, nine failed elaborations, 5-seed confirmation (did not generalise), horizon-dependence finding |
| `renewal-net-design.md` | Renewal lab record: design, corrections, final verdict (0/45 on accuracy; interpretability earned) |
| `adversarial-priority-check.md` | Priority audit of novelty claims + verified bibliography corrections |
| `preprocessing-audit.md` | Leakage audit (clean); protocol facts both papers must state |
| `paper-c-design.md` | Paper C design spec, with changelog |
| `EpiSIG-Net-v3-Design.md` | Parked architecture line (results predate the protocol fix) |
| `../program.md` | Both autoresearch programmes, CLOSED, kept for the record |

## Superseded

`paper-renewal-interpretability.md` — Paper A's content outline. The LaTeX in
`plos-renewal/` is now the live artefact and carries newer numbers (5-seed
kernel statistics, the naive-baseline table, the Wallinga–Lipsitch reframe).

## `archive/` — do not cite

Pre-protocol-fix analyses, the AIIM submission bundle, and stale results CSVs.
Every comparison in them uses the invalid pooled h…2h−1 baseline scoring
(ledger E1), and most use a 50/20/30 split rather than the current 60/20/20.
`paper_results_final.csv` is the provenance of the AIIM manuscript's table —
kept as evidence, not as data.

## `GNN forecasting/`

A Zotero export of eight literature PDFs. Not source; read and summarised in
`audit-2026-08-24/04-user-papers-folder.md`.
