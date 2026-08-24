# 06 — Data provenance

Recovered 25 Aug 2026 by sweeping the user's other repositories under
`c:\Users\ajaoo\Documents\GitHub\`. The three `.txt` files are byte-identical
(md5) across `MSAGAT-Net`, `colagnn` and `EpiGNN` — confirmed independently
here and at `doc/preprocessing-audit.md:42-43`.

Repositories that contributed: **EpiHealthForecast**, **Multivate-forecasting**,
**colagnn**, **EpiGNN**. The literal filenames `nhs_timeseries.txt`,
`ltla_timeseries.txt`, `australia-covid.txt` exist nowhere else.

---

## 1. NHS-ICUBeds — `nhs_timeseries.txt` (7 × 895) — **verified numerically**

**URL / API**

```
https://api.coronavirus.data.gov.uk/v2/data
  ?areaType=nhsRegion
  &areaCode=<code>
  &metric=covidOccupiedMVBeds
  &format=csv
```

one call per region. Evidence:
`EpiHealthForecast/notebooks/Experiments/data_collection.ipynb:107-113`
(literal URL list); duplicated verbatim in
`Multivate-forecasting/scripts/data/make_dataset.py:184-190`. A second access
path uses the official `uk_covid19` client:
`Cov19API(filters=["areaType=nhsRegion", f"areaName={region}"], structure=NHS_STRUCTURE)`
looping over the same 7 regions in the same order
(`EpiHealthForecast/scripts/data_collection.py:130-138`).

Region codes: E40000007 East of England, E40000003 London, E40000008
Midlands, E40000009 North East and Yorkshire, E40000010 North West,
E40000005 South East, E40000006 South West.

**Metric: `covidOccupiedMVBeds`** — mechanical-ventilation bed occupancy;
this is what the manuscript calls "NHS-ICUBeds".

*Proof by reconstruction.* `EpiHealthForecast/data/processed/merged_nhs_covid_data.csv`
has `East of England, 2020-04-01, 0.0` then `2020-04-02, 119.0`. The 7-day
trailing (expanding-window) mean of (0, 119) = **59.5**, which is exactly
`nhs_timeseries.txt` row 2, column 1. London's (0, 673) → **336.5** matches
row 2 column 2 identically. This value-for-value match rules out
`hospitalCases` and `newAdmissions`.

**Column order (1–7):** East of England, London, Midlands, North East and
Yorkshire, North West, South East, South West. Verified by the reconstruction
above and corroborated by `data/geo/nhs_centroids.csv` (alphabetical) with its
provenance note at `src/scripts/build_adjacency.py:12-13`: "ONS NHS England
Regions (April 2021) centroids in alphabetical order; at 150 km this
reproduces `data/nhs-adj.txt` exactly."

**Date range: 2020-04-01 → 2022-09-12** (895 daily rows). Derived from the
merged CSV's shape (6265 rows / 7 regions = 895 exactly) and its min/max date;
2020-04-01 + 894 days = 2022-09-12.

**Preprocessing:** causal/trailing 7-day mean, expanding window for the first
six days (denominators 1,2,…,7,7,7…). No clipping, no region merging — all 7
official NHS England regions used as-is.

---

## 2. LTLA-COVID — `ltla_timeseries.txt` (372 × 839)

**URL / API:** the same UKHSA service with `areaType=ltla`, accessed through
`uk_covid19.Cov19API`. Evidence: `EpiHealthForecast/scripts/data.py:52-95`
(`get_uk_df(area_type="ltla", …)`, mapping `dailyCases →
newCasesBySpecimenDate` explicitly at `:71`). Column headers in the raw pulls
`EpiHealthForecast/data/data_others/Other_data/ltla_2021-12-31.csv` and
`ltla_2023-05-18.csv` confirm the field name and area type.

**Metric: `newCasesBySpecimenDate`** — daily confirmed cases by specimen date.

**Area list and order:** 372 UK Lower Tier Local Authorities = ONS Local
Authority Districts (December 2021, UK, BUC) **minus City of London and Isles
of Scilly**, which UKHSA reporting merges into Hackney and Cornwall
respectively, name-sorted. Source: `src/scripts/build_adjacency.py:6-11`.
`data/geo/ltla_centroids.csv` has exactly 372 rows, alphabetically sorted,
starting "Aberdeen City". Corroborated by
`EpiHealthForecast/data/raw/Local_Authority_Districts_December_2022_…csv`
(374 LADs − 2 merged pairs = 372).

**Caveat, stated for honesty:** an attempt to confirm column 1 = Aberdeen City
by cross-matching real per-area daily counts from
`EpiHealthForecast/data/raw/ltla_data.csv` did **not** succeed — the
leading-zero run in column 1 did not line up with Aberdeen's first-case date
under any tested start date. The *area count* and *composition* are solid; the
exact *column order* rests on the repository's own (self-described
"recovered") provenance note. `doc/MSAGAT-RESEARCH-LEDGER.md:125` independently
flags LTLA appearing as both 307 and 372 nodes across working documents.

**Date range: 2020-04-01 → 2022-07-18 (839 days) — inferred, not read.**
The file has no header or date column. The range was established by an
epidemic-wave signature match on the row-sums across all 372 columns:

| row | row-sum | implied date | UK reality |
|---|---|---|---|
| 1 | ~2471 | 2020-04-01 | tail of the first wave, declining into lockdown |
| 60 | ~1154 | 2020-05-30 | continued decline |
| 270 | 55,370 | 2020-12-27 | Alpha surge, ~50–60k/day |
| 630 | 196,458 | 2021-12-22 | Omicron, peak ~190–200k/day |
| 660 | 188,707 | 2022-01-21 | Omicron plateau |

The Omicron peak's magnitude and timing are distinctive enough to make this a
strong circumstantial match, but the paper should say **"inferred"** or the
series should be re-pulled to confirm.

**Preprocessing:** the same causal 7-day trailing mean (same expanding
denominator signature). Region merging happens upstream at UKHSA — the raw
data already reports "Hackney and City of London" and "Cornwall and Isles of
Scilly" as single entities, so no merge step was performed locally.

---

## 3. Australia-COVID — `australia-covid.txt` (8 × 556)

**Not ours.** It is an unmodified file from the public **EpiGNN** release.
`EpiGNN` is a fork (`upstream = https://github.com/Xiefeng69/EpiGNN.git`) and
`data/australia-covid.txt` was added in the upstream author's own `init`
commit **`9560aee7d779a9da63609ce32dbe1a3f3480a912`, Xiefeng69, 2022-06-15**.
The `colagnn` fork picked up a byte-identical copy later (commit `3efa2f0`,
2025-03-21).

**Source:** `EpiGNN/README.md:17` — "the COVID-related data is publicly
avaliable at [JHU-CSSE](https://github.com/CSSEGISandData/COVID-19)." No
per-file detail is given.

**Metric / preprocessing:** raw integer daily new-case counts, unsmoothed —
consistent with a first difference of JHU's cumulative confirmed series.
Confirmed unsmoothed at `doc/preprocessing-audit.md:19`.

**Date range and the 8-column state order: NOT determined.** An attempt to
reconstruct them from `colagnn/data/time_series_covid19_confirmed_global.csv`
failed — that snapshot covers only 2020-01-22 to 2021-12-31 (434 days) and is
evidently a different, later download (added March 2025 for the Spain dataset).
The final rows reach ~9987, consistent in magnitude with Australia's Omicron
wave (Jan 2022), implying an end around Dec 2021–Jan 2022 and a start in
mid-to-late 2020 — but this is an unverified inference.

**How to cite it:** "the Australia-COVID series distributed with EpiGNN
(Xie et al., 2022), derived from JHU CSSE." **Do not assert a calendar range.**

---

## 4. What could not be determined

1. **Exact access/download timestamps** for the NHS and LTLA pulls. No
   repository preserves a raw response with the matching end date; git and
   file mtimes reflect repo-import dates (Jul 2025), not the original
   download. Given UKHSA's 1–3 day reporting lag, the pulls most likely
   occurred shortly after each series' end date. The sibling snapshots in
   `EpiHealthForecast/data/data_others/Other_data/` follow a convention where
   the filename date is one day after the last data date, so by that
   convention the LTLA pull happened within days of 2022-07-18 — but neither
   surviving snapshot is the actual source.
   **Recommended wording:** "downloaded from the UKHSA COVID-19 dashboard API
   (`api.coronavirus.data.gov.uk`); the NHS series ends 2022-09-12 and the
   LTLA series 2022-07-18." Do not invent an access date.
2. **The exact column order of the 372 LTLA areas** (see the caveat above).
3. **Australia's calendar range and 8-column state order.**
4. **Whether negative specimen-date revisions were clipped** in the LTLA
   series — plausible, since specimen-date series are revised and can go
   negative on backfill, but not confirmed.
5. **The generation script itself.** Despite strong numeric evidence tying
   these files to the `EpiHealthForecast` pipeline, the script or notebook
   cell that pivoted the long-format data into the wide, smoothed matrices and
   wrote `nhs_timeseries.txt` / `ltla_timeseries.txt` was not found.

---

## 5. Consequence for Paper C

**The raw, unsmoothed sources exist**:

- `EpiHealthForecast/data/processed/merged_nhs_covid_data.csv` (long format,
  6265 rows, integer occupancy counts)
- `EpiHealthForecast/data/raw/ltla_data.csv`

So Paper C's negative-binomial / zero-inflated count likelihood can be applied
to genuine integer counts rather than to 7-day means. Paper B keeps the
existing smoothed matrices unchanged, so nothing already run is invalidated
and the two papers remain comparable through the shared protocol description.
