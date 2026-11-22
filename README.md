# Inflation aversion and asymmetric consumer sentiment responses: evidence from the Michigan Surveys of Consumers

Replication code and unrounded outputs for the manuscript and its Supplementary Materials
(Michigan Surveys of Consumers, 78,097 six-month interview pairs, 1980-2026).

## Contents
- `scripts/` - all data-processing and estimation code, numbered in running order.
- `output/` - unrounded results (CSV/JSON) behind every table and number; nothing in `output/` is rounded.
- `refs/` - Crossref check of the reference list.
- `data/` - codebook, variable list and instructions to obtain the raw data (see `data/README.md`).

## Requirements
Python 3.12.6, NumPy 2.3.2, pandas 2.2.3, statsmodels 0.14.6, matplotlib 3.10.1
(`pip install -r requirements.txt`). Steps 2 and 4 estimate models with about 550 survey-month fixed
effects and need about 4 GB of free RAM.

## Data
The SDA extract (5 September 2026) and the linked panel are included under `data/`. `data/README.md` explains how to obtain the original microdata from SDA.

## Order of execution (run from the repository root)
| Step | Command | Produces |
|---|---|---|
| 1 | `python scripts/01_clean_panel.py` | `data/processed/panel_clean.csv` |
| 2 | `python scripts/06_revision_analysis.py` | sample counts, SD of ICS (S4.1) -> `output/revision/` |
| 3 | `python scripts/07_subsample_diagnostics.py`, `python scripts/08_logic_checks.py` | diagnostics by mode, decade, size of revision -> `output/revision/` |
| 4 | `python scripts/09_final_estimates.py` | Tables 1-2, S1-S5 -> `output/final/` (30-60 min, bootstrap) |
| 5 | `python scripts/revise01/run1_tables.py level` | magnitude restriction, curvature / initial expectations, common support + overlap weights -> `output/revise01/` |
| 6 | `python scripts/revise01/run2_bootstrap.py level 2000 2000` | marginal slopes and full-pipeline overlap bootstrap -> `output/revise01/` |
| 7 | `python scripts/10_final_figures.py` | Fig. 1, Figs. S1-S6 -> `output/figures/final/` |
| 8 | `python scripts/11_verify_references.py` | `refs/reference_check.csv` |

The scripts in `scripts/revise01/` also accept `log` instead of `level` (revision measured as
100 x dln(1 + PX1/100)); this is a scale sensitivity check reported in `output/revise01/SO_SANH_LEVEL_VS_LOG.md`
and is not used in the main text.

## Seed and bootstrap configuration
- Seed **42** in every script.
- **2,000** respondent-cluster bootstrap draws (respondents resampled with replacement; both pairs of a respondent
  are drawn together) for: the 95% CI of lambda (`09_final_estimates.py`), the marginal-slope contrasts,
  and the full-pipeline overlap bootstrap (common support, logit, overlap weights and the model are
  re-estimated in every draw).
- `run2_bootstrap.py` uses a multiprocessing pool; set the number of workers with the environment
  variable `NPROC` (default 12; use 4-6 on machines with less than 16 GB of RAM).
- Standard errors are clustered by respondent. Confidence intervals are percentile intervals.

## Where each result is
| Result | File |
|---|---|
| Table 1 | `output/final/table1_main.csv` |
| Table 2 | `output/final/table2_robustness.csv` |
| Table S1 (sample flow) | `output/final/final_results.json` ("flow") |
| Table S2 (descriptives) | `output/final/tableS_descriptives.csv` |
| Table S3 | `output/final/tableS_main_full_coefficients.csv` |
| Table S4 / S5 | `output/final/tableS_sources.csv`, `tableS_additional.csv` |
| S6 curvature and initial expectations | `output/revise01/t2_nonlinear_level.csv` |
| S7 size of revision (web) | `output/revise01/t1_magnitude_level.csv` |
| S9.1 marginal slopes and contrasts | `output/revise01/t4_marginal_slopes_level.csv`, `t4_contrasts_level.csv` |
| S10 common support, overlap weights, balance, bootstrap | `t3_mode_level.csv`, `t3_balance_level.csv`, `t5_overlap_bootstrap_level.json`, `t5_overlap_draws_level.csv` |
| Figures | `output/figures/final/` |

In the paper, lambda is the *inflation aversion asymmetry ratio* |beta_up|/|beta_down| (a descriptive ratio of sentiment slopes, not a loss aversion coefficient).

Headline unrounded numbers: pooled lambda 2.325 (95% CI [1.701, 3.397]); telephone 1.407; web 2.870.
