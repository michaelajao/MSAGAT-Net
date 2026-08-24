# Audit of 24–25 August 2026

Five reads were run over the repository, its results, and the literature
before the three-paper programme was planned. This directory is their
verbatim record. Nothing in it is a plan; it is evidence.

| File | What it contains |
|---|---|
| [01-docs-and-results-digest.md](01-docs-and-results-digest.md) | Every `.md` in the repo plus every result CSV: findings E1–E12, the dead claims, the current DM grid, calibration numbers, and 13 contradictions between documents |
| [02-architecture-audit.md](02-architecture-audit.md) | `models.py` / `train.py` / `data.py` / `evaluate.py` / `utils.py` read line by line against three trained checkpoints: module-by-module forward pass, parameter formula, what is inert vs live, code-vs-manuscript mismatches |
| [03-literature-baselines-and-followons.md](03-literature-baselines-and-followons.md) | Cola-GNN, EpiGNN, MepoGNN, STAN, CNNRNN-Res, DCRNN, LSTNet — mechanisms, protocols, reported numbers, verbatim future work — plus the 2023–2026 follow-ons and benchmark papers |
| [04-user-papers-folder.md](04-user-papers-folder.md) | The eight PDFs in `doc/GNN forecasting/` plus Fritz et al. 2022: mechanism, data requirements, protocol, limitations, and what each leaves open |
| [05-provenance-audit.md](05-provenance-audit.md) | How every result, prediction, checkpoint, table and figure is produced and saved; the writer table; 13 confirmed defects; the figure inventory |
| [06-data-provenance.md](06-data-provenance.md) | Where `nhs_timeseries.txt`, `ltla_timeseries.txt` and `australia-covid.txt` actually came from, recovered from the user's other repositories |

## The three conclusions these reads produced

1. **The architecture question is closed.** MSAGAT-Net's components are
   Cola-GNN's own three blocks (§3.2–3.4 of that paper), re-used by EpiGNN
   and republished as STTGNN in *Knowledge-Based Systems* (April 2026). See
   file 03.
2. **The implemented model never ran the idea.** No multi-scale temporal
   convolution exists in the code; the learnable graph bias and the hop
   fusion weights are numerically zero in every trained checkpoint. See
   file 02.
3. **That degenerate model still ties the field** — 74 of 103 Diebold-
   Mariano comparisons are ties, all 10 losses are Australia. This is the
   same result SpatialEpiBench (May 2026) reports across 11 datasets. See
   files 01 and 03.

Together these move the work from an architecture contribution to an
evaluation contribution, which is what Paper B now is.
