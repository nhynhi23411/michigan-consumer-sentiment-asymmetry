# Data access

The extract used in the paper is included in this repository (`data/raw/sca_microdata_raw.csv`, `data/processed/panel_clean.csv`, `data/analysis_variables/`). The underlying microdata are public and free from the University of Michigan Surveys of Consumers; to rebuild them from the original source:

1. Open the SDA Archive: https://sda.umsurvey.org (Surveys of Consumers public microdata, 1978-2026;
   see also https://data.sca.isr.umich.edu).
2. Use the *customized subset* tool to extract **all 44 variables** for all records. The extract used in the
   paper was produced on **5 September 2026** (342,345 records, January 1978 to February 2026).
   The codebook of this extract is `data/raw/sca_microdata_codebook.txt`.
3. Save it as CSV at `data/raw/sca_microdata_raw.csv`.
4. Run `python scripts/01_clean_panel.py`; it writes `data/processed/panel_clean.csv` (109,828 linked six-month pairs).

The 15 variables actually used are listed in `data/analysis_variables/analysis_variables_list.csv`
(YYYYMM, ID, IDPREV, DATEPR, METHOD, ICS, ICC, ICE, PX1, PX5, AGE, INCOME, EDUC, REGION, WT).

`data/external/` holds the published monthly Michigan series (www.sca.isr.umich.edu, downloaded
30 September 2026), used only for Fig. S2.
Later SDA extracts may contain additional months and therefore will not reproduce the numbers exactly.
