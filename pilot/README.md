# Illustrative audits of public crash databases

Scripts, derived tables and run logs for the post hoc audits in Section 6 of the article. The audits illustrate diagnostic signatures; they do not calibrate or validate the simulation mechanism, and their values are single-audit point estimates rather than population estimates. Per-dataset results are reported in Supplementary Tables S22–S33.

Logs record the original runs; local path prefixes have been removed from them.

## Rerunning an audit

Run each script from its own directory after placing the raw inputs where the table below indicates. Dependencies: numpy, pandas, scipy, scikit-learn, pyreadstat (NMVCCS) and odfpy (the DfT `.ods` table).

| Directory | Audit | Scripts | Raw inputs |
|---|---|---|---|
| `stats19_2024/` | STATS19 2024 single-year audit | `audit.py` | `data/dft-road-casualty-statistics-collision-2024.csv` and `data/dft-road-casualty-statistics-vehicle-2024.csv` (not included) |
| `stats19_ibrs/` | Injury-based versus traditional severity recording within 2024 | `ibrs_analysis.py` | reads `../stats19_2024/data/` |
| `stats19_multiyear/` | 2016 versus 2024 regime contrast; cross-period transfer and intercept repair (`transfer/`) | `analysis.py`, `transfer/transfer_analysis.py` | `collision_2016.csv` (included, see below) and `collision_2024.csv`, a copy of the 2024 collision file above (not included) |
| `dft_cf_fatal_2021/` | Roadside versus post-investigation contributory factors in 2021 fatal collisions | `extract.py`, then `build_report.py` | `ras_cfs_final.ods` (included) |
| `crss/` | CRSS 2024, unweighted (`analysis.py`) and with the CRSS analysis weight (`weighted_analysis.py`) | `analysis.py`, `weighted_analysis.py` | `CRSS2024/CRSS2024CSV/` (NHTSA CRSS 2024 CSV release; not included) |
| `nmvccs/` | NMVCCS 2005–2007, folds grouped by crash | `analysis.py` | `raw/pub_extracted/` (public SAS files and `formats.sas7bcat`) and `raw/interviews_extracted/` (interview SAS files); not included |
| `fars_vs_crss/` | Exploratory FARS–CRSS fatal-case contrast, 2022 | `audit.py` | `fars2022/FARS2022NationalCSV/` and `crss2022/CRSS2022CSV/` (not included) |

`build_report.py` assembles an internal narrative report from the derived tables; that report is not part of this release. `stats19_multiyear/download_2016.py` is a resumable downloader for the DfT 1979–latest collision file; the 2016 table used in the audit is the archived single-year file described below.

## Included third-party files

- `stats19_multiyear/collision_2016.csv`: the 2016 STATS19 collision table (136,621 records, the count in the official 2016 release), taken from the archived official single-year 2016 file on data.dft.gov.uk (Internet Archive snapshot of 23 May 2023) with columns renamed to the 2024 collision-table names. It is included because the single-year 2016 file is no longer served by the DfT site.
- `dft_cf_fatal_2021/ras_cfs_final.ods`: the data table published with the DfT statistics release on contributory factors in fatal collisions (2021 data).

Both contain public sector information licensed under the Open Government Licence v3.0.

## Sources

- STATS19 road safety open data: https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
- Contributory factors in fatal collisions, 2021 data: https://www.gov.uk/government/statistics/contributory-factors-in-fatal-collisions-comparing-stats19-with-post-investigation-recording-2021-data
- Crash Report Sampling System (CRSS): https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system
- Fatality Analysis Reporting System (FARS): https://www.nhtsa.gov/crash-data-systems/fatality-analysis-reporting-system
- National Motor Vehicle Crash Causation Survey (NMVCCS): https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/NASS/NMVCCS/
