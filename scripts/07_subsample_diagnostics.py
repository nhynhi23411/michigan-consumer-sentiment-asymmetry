"""
Diagnostics requested during the September 2026 revision: where does the asymmetry
come from? Fully interacted model (as in 06_revision_analysis.py) estimated by
collection mode, by decade of the first interview, and with survey-month fixed effects
within the telephone period. Respondent-clustered standard errors.
Output: output/revision/table_subsamples.csv
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
p = pd.read_csv("data/processed/panel_clean.csv")
is_second = p["IDPREV_t1"].notna()
p["person"] = (np.where(is_second, p["IDPREV_t1"], p["ID_t1"]).astype("int64").astype(str) + "_" +
               np.where(is_second, p["DATEPR_t1"], p["YYYYMM_t1"]).astype("int64").astype(str))
ok = p.EDUC_t1.between(1, 6) & p.REGION_t1.between(1, 4) & p.AGE_t1.between(18, 97) & (p.INCOME_t1 >= 0)
d = p[(p.delta_px1_w != 0) & ok].copy()
d["up"] = (d.delta_px1_w > 0).astype(float)
d["income10k"] = d.INCOME_t1 / 1e4
for k in (2, 3, 4):
    d[f"r{k}"] = (d.REGION_t1 == k).astype(float)
CTRL = ["AGE_t1", "income10k", "EDUC_t1", "r2", "r3", "r4"]


def est(df, label, fe=None):
    cols = {"_c": np.ones(len(df)), "x": df.delta_px1_w.values}
    for c in CTRL:
        cols[c] = df[c].values
    X = pd.DataFrame(cols, index=df.index)
    for c in list(X.columns):
        X[c + "_u"] = X[c] * df.up.values
    if fe is not None:
        X = pd.concat([X, pd.get_dummies(df[fe], prefix="m", drop_first=True, dtype=float)], axis=1)
    r = sm.OLS(df.delta_sentiment_w, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.person)[0]})
    bd, g = r.params["x"], r.params["x_u"]
    return {"label": label, "n": len(df), "n_up": int(df.up.sum()), "b_down": bd, "se_down": r.bse["x"],
            "b_up": bd + g, "gamma": g, "se_gamma": r.bse["x_u"], "p_gamma": r.pvalues["x_u"],
            "lambda": abs(bd + g) / abs(bd), "first": int(df.YYYYMM_t1.min()), "last": int(df.YYYYMM_t2.max())}


rows = []
web = (d.METHOD_t1 == 2) & (d.METHOD_t2 == 2)
rows.append(est(d[web], "Web pairs only (2024-2026)"))
rows.append(est(d[~web], "Telephone pairs only (1980-2024)"))
rows.append(est(d[~web], "Telephone pairs, survey-month FE", fe="YYYYMM_t2"))
rows.append(est(d[web], "Web pairs, survey-month FE", fe="YYYYMM_t2"))
dec = (d.YYYYMM_t1 // 1000) * 10  # 198, 199 ... -> decade
for lo, hi, lab in [(198000, 198912, "1980s"), (199000, 199912, "1990s"), (200000, 200912, "2000s"),
                    (201000, 201912, "2010s"), (202000, 202312, "2020-2023 (telephone)")]:
    sub = d[(d.YYYYMM_t1 >= lo) & (d.YYYYMM_t1 <= hi) & ~web]
    rows.append(est(sub, f"First interview in {lab}"))
out = pd.DataFrame(rows)
out.to_csv("output/revision/table_subsamples.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(4).to_string())
