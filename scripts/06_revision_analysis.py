"""
Revision analysis (September 2026), following the supervisor's revision guide
(Huong_dan_hoan_thien_ban_thao_Economics_Letters.docx).

What this script fixes relative to 02_analysis_main.py / 03_robustness.py:
  * Main estimate: ONE specification delivers the two slopes, the slope difference,
    lambda and its CI. We use the fully interacted pooled model (every regressor,
    including the intercept and all controls, interacted with the increase indicator),
    which reproduces the two separate regressions exactly and lets gamma be tested in
    the same model that delivers lambda.
  * The restricted pooled model of the old Table 3 (common control coefficients) is
    kept as an alternative; its implied lambda (about 2.53) is reported separately and
    is NOT presented as the same quantity as the separate-regression lambda.
  * Standard errors: HC1 and respondent-clustered (a respondent can contribute two
    consecutive pairs). CI for lambda from a respondent-cluster bootstrap (2,000
    draws, seed 42) and from the delta method.
  * Robustness table = baseline + seven checks. The weighted row now uses weights in
    the interaction model too (old code used an unweighted interaction model).
    The old "median split" row is relabelled: the median of the change is exactly 0,
    so that row simply adds the zero-change pairs to the lower group.
    PX5, ICC and ICE changes are winsorized at 1/99 like the baseline variables.
  * Sample accounting: pairs, unique respondents, exact first/last months.
  * ICS standard deviation reported at three levels; monthly PX1 series rebuilt with
    missing codes removed, as weighted median and weighted mean.
Outputs go to output/revision/.
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

RAW = "data/raw/sca_microdata_raw.csv"
PANEL = "data/processed/panel_clean.csv"
OUT = "output/revision"
os.makedirs(OUT, exist_ok=True)
SEED = 42
NBOOT = 2000
PX_MISSING_MIN = 96  # codebook: -97 and 96-* are missing


def winsor(s, lo=0.01, hi=0.99):
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


def clean_px(s):
    return s.where((s != -97) & (s < PX_MISSING_MIN))


log = {}

# ------------------------------------------------------------------ sample accounting
raw = pd.read_csv(RAW, skipinitialspace=True)
raw.columns = [c.strip() for c in raw.columns]
for c in raw.columns:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
log["raw_records"] = int(len(raw))
log["raw_first_month"] = int(raw.YYYYMM.min())
log["raw_last_month"] = int(raw.YYYYMM.max())
log["raw_reinterview_records"] = int(raw.IDPREV.notna().sum())

p = pd.read_csv(PANEL)
log["linked_pairs_with_valid_dICS_dPX1"] = int(len(p))

# respondent identifier: root = first interview of the person
is_second = p["IDPREV_t1"].notna()
p["root_id"] = np.where(is_second, p["IDPREV_t1"], p["ID_t1"])
p["root_month"] = np.where(is_second, p["DATEPR_t1"], p["YYYYMM_t1"])
p["person"] = p["root_id"].astype("int64").astype(str) + "_" + p["root_month"].astype("int64").astype(str)

p["educ_ok"] = p["EDUC_t1"].between(1, 6)
p["region_ok"] = p["REGION_t1"].between(1, 4)
p["age_ok"] = p["AGE_t1"].between(18, 97)
p["income_ok"] = p["INCOME_t1"].notna() & (p["INCOME_t1"] >= 0)
controls_ok = p.educ_ok & p.region_ok & p.age_ok & p.income_ok

log["pairs_zero_change_raw_dPX1"] = int((p.delta_px1 == 0).sum())
log["pairs_zero_change_winsorized"] = int((p.delta_px1_w == 0).sum())

d = p[(p.delta_px1_w != 0) & controls_ok].copy()
d["up"] = (d.delta_px1_w > 0).astype(int)
d["income10k"] = d["INCOME_t1"] / 10000.0
d["web_pair"] = (d.METHOD_t1 == 2) | (d.METHOD_t2 == 2)
d["web_switch"] = (d.METHOD_t1 == 2) != (d.METHOD_t2 == 2)
d["mode_switch_any"] = d.METHOD_t1 != d.METHOD_t2
log["analytic_pairs"] = int(len(d))
log["analytic_increase"] = int(d.up.sum())
log["analytic_decrease"] = int((1 - d.up).sum())
log["analytic_unique_respondents"] = int(d.person.nunique())
log["respondents_with_two_pairs"] = int((d.person.value_counts() == 2).sum())
log["respondents_with_more_than_two_pairs"] = int((d.person.value_counts() > 2).sum())
log["first_interview_first_month"] = int(d.YYYYMM_t1.min())
log["first_interview_last_month"] = int(d.YYYYMM_t1.max())
log["second_interview_first_month"] = int(d.YYYYMM_t2.min())
log["second_interview_last_month"] = int(d.YYYYMM_t2.max())
log["pairs_any_method_switch"] = int(d.mode_switch_any.sum())
log["pairs_phone_web_switch"] = int(d.web_switch.sum())
log["pairs_involving_web"] = int(d.web_pair.sum())
log["dropped_missing_controls"] = int(((p.delta_px1_w != 0) & ~controls_ok).sum())

# ------------------------------------------------------------------ design matrices
REG = ["r2", "r3", "r4"]
for k in (2, 3, 4):
    d[f"r{k}"] = (d.REGION_t1 == k).astype(float)
BASE_X = ["AGE_t1", "income10k", "EDUC_t1"] + REG


def design(df, x, extra=None):
    """Fully interacted design: const, x, controls, and each times up."""
    cols = ["_c", x] + BASE_X + (extra or [])
    X = pd.DataFrame({"_c": 1.0}, index=df.index)
    X[x] = df[x]
    for c in BASE_X + (extra or []):
        X[c] = df[c]
    for c in cols:
        X[f"{c}__up"] = X[c] * df["up"]
    return X


def fit_full(df, y, x, w=None, cluster=True):
    X = design(df, x)
    if w is None:
        m = sm.OLS(df[y], X)
    else:
        m = sm.WLS(df[y], X, weights=df[w])
    if cluster:
        r = m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["person"])[0]})
    else:
        r = m.fit(cov_type="HC1")
    b_down = r.params[x]
    gamma = r.params[f"{x}__up"]
    b_up = b_down + gamma
    V = r.cov_params()
    var_up = V.loc[x, x] + V.loc[f"{x}__up", f"{x}__up"] + 2 * V.loc[x, f"{x}__up"]
    lam = abs(b_up) / abs(b_down)
    # delta method for lambda = b_up / b_down (both same sign)
    g = np.zeros(len(r.params))
    idx = list(r.params.index)
    # lambda = (b_down + gamma)/b_down ; d/d b_down = -gamma/b_down^2 ; d/d gamma = 1/b_down
    g[idx.index(x)] = -gamma / b_down ** 2
    g[idx.index(f"{x}__up")] = 1 / b_down
    se_lam = float(np.sqrt(g @ V.values @ g))
    return {
        "b_up": float(b_up), "se_up": float(np.sqrt(var_up)),
        "b_down": float(b_down), "se_down": float(r.bse[x]),
        "gamma": float(gamma), "se_gamma": float(r.bse[f"{x}__up"]), "p_gamma": float(r.pvalues[f"{x}__up"]),
        "lambda": float(lam), "se_lambda_delta": se_lam,
        "lambda_ci_delta": [float(lam - 1.96 * se_lam), float(lam + 1.96 * se_lam)],
        "n_up": int(df.up.sum()), "n_down": int((1 - df.up).sum()), "n": int(len(df)),
        "n_resp": int(df.person.nunique()),
        "r2_full": float(r.rsquared),
        "_res": r,
    }


def cluster_bootstrap_lambda(df, y, x, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    X = design(df, x).values
    Y = df[y].values
    persons, inv = np.unique(df["person"].values, return_inverse=True)
    rows_by = pd.Series(np.arange(len(df))).groupby(inv).apply(np.array).values
    ix = list(design(df.head(2), x).columns)
    jx, jg = ix.index(x), ix.index(f"{x}__up")
    lams, gams = [], []
    for _ in range(nboot):
        pick = rng.integers(0, len(persons), len(persons))
        rows = np.concatenate(rows_by[pick])
        beta, *_ = np.linalg.lstsq(X[rows], Y[rows], rcond=None)
        lams.append(abs(beta[jx] + beta[jg]) / abs(beta[jx]))
        gams.append(beta[jg])
    lams = np.array(lams)
    return lams, np.array(gams)


results = {}

# ------------------------------------------------------------------ main specification
main = fit_full(d, "delta_sentiment_w", "delta_px1_w")
main_hc1 = fit_full(d, "delta_sentiment_w", "delta_px1_w", cluster=False)
lams, gams = cluster_bootstrap_lambda(d, "delta_sentiment_w", "delta_px1_w")
main["lambda_ci_boot"] = [float(np.percentile(lams, 2.5)), float(np.percentile(lams, 97.5))]
main["lambda_boot_se"] = float(lams.std(ddof=1))
main["boot_share_lambda_le_1"] = float((lams <= 1).mean())
main["gamma_ci_boot"] = [float(np.percentile(gams, 2.5)), float(np.percentile(gams, 97.5))]
pd.Series(lams).to_csv(f"{OUT}/bootstrap_lambda_draws.csv", index=False, header=["lambda"])
results["main_cluster"] = main
results["main_hc1"] = main_hc1

# full coefficient table for the main model (controls by group)
r = main["_res"]
rows = []
for c in ["_c", "delta_px1_w"] + BASE_X:
    b_d = r.params[c]
    b_i = r.params[f"{c}__up"]
    V = r.cov_params()
    se_u = np.sqrt(V.loc[c, c] + V.loc[f"{c}__up", f"{c}__up"] + 2 * V.loc[c, f"{c}__up"])
    rows.append({"term": c, "decrease_b": b_d, "decrease_se": r.bse[c],
                 "increase_b": b_d + b_i, "increase_se": se_u,
                 "diff_b": b_i, "diff_se": r.bse[f"{c}__up"], "diff_p": r.pvalues[f"{c}__up"]})
pd.DataFrame(rows).to_csv(f"{OUT}/table_main_full_coefficients.csv", index=False)

# separate regressions check (must equal the fully interacted slopes)
sep = {}
for g_ in (1, 0):
    sub = d[d.up == g_]
    m = smf.ols("delta_sentiment_w ~ delta_px1_w + AGE_t1 + income10k + EDUC_t1 + r2 + r3 + r4", data=sub).fit(cov_type="HC1")
    sep["up" if g_ else "down"] = {"b": float(m.params["delta_px1_w"]), "se_hc1": float(m.bse["delta_px1_w"]),
                                    "r2": float(m.rsquared), "n": int(m.nobs)}
results["separate_regressions_hc1"] = sep

# restricted pooled model (old Table 3, column 3): common control coefficients
rp = smf.ols("delta_sentiment_w ~ delta_px1_w * up + AGE_t1 + income10k + EDUC_t1 + r2 + r3 + r4", data=d).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(d["person"])[0]})
bd, gm = rp.params["delta_px1_w"], rp.params["delta_px1_w:up"]
V = rp.cov_params()
gvec = np.zeros(len(rp.params)); ids = list(rp.params.index)
gvec[ids.index("delta_px1_w")] = -gm / bd ** 2
gvec[ids.index("delta_px1_w:up")] = 1 / bd
se_l = float(np.sqrt(gvec @ V.values @ gvec))
results["restricted_pooled"] = {
    "b_down": float(bd), "gamma": float(gm), "se_gamma": float(rp.bse["delta_px1_w:up"]),
    "p_gamma": float(rp.pvalues["delta_px1_w:up"]), "b_up": float(bd + gm),
    "lambda": float(abs(bd + gm) / abs(bd)), "lambda_ci_delta": [float(abs(bd + gm) / abs(bd) - 1.96 * se_l),
                                                                 float(abs(bd + gm) / abs(bd) + 1.96 * se_l)],
    "increase_indicator": float(rp.params["up"]), "se_increase_indicator": float(rp.bse["up"]),
    "r2": float(rp.rsquared), "n": int(rp.nobs)}

# ------------------------------------------------------------------ robustness: baseline + seven checks
rob = []


def add(label, df, y="delta_sentiment_w", x="delta_px1_w", w=None):
    res = fit_full(df, y, x, w=w)
    res.pop("_res")
    res["label"] = label
    rob.append(res)
    return res


add("Baseline (Table 1)", d)
add("Excluding pairs with a change in collection mode", d[~d.mode_switch_any])
add("Excluding first interviews in 2020-2022", d[(d.YYYYMM_t1 < 202001) | (d.YYYYMM_t1 > 202212)])

# zero-change pairs added to the lower group (the old 'median split'; median of dPX1 is 0)
z = p[controls_ok].copy()
z["up"] = (z.delta_px1_w > 0).astype(int)
z["income10k"] = z["INCOME_t1"] / 10000.0
for k in (2, 3, 4):
    z[f"r{k}"] = (z.REGION_t1 == k).astype(float)
log["median_dPX1_w_all_valid_pairs"] = float(p.delta_px1_w.median())
add("Zero-change pairs added to the non-increase group", z)

# five-year horizon
q = p[controls_ok].copy()
q["PX5_t1c"], q["PX5_t2c"] = clean_px(q.PX5_t1), clean_px(q.PX5_t2)
q["dpx5"] = q.PX5_t2c - q.PX5_t1c
q = q.dropna(subset=["dpx5"])
q["dpx5_w"] = winsor(q["dpx5"])
q = q[q.dpx5_w != 0].copy()
q["up"] = (q.dpx5_w > 0).astype(int)
q["income10k"] = q["INCOME_t1"] / 10000.0
for k in (2, 3, 4):
    q[f"r{k}"] = (q.REGION_t1 == k).astype(float)
log["px5_first_month_in_pairs"] = int(q.YYYYMM_t1.min())
add("Five-to-ten-year expectations (PX5) instead of PX1", q, x="dpx5_w")

# sub-indices, winsorized like the baseline
d["dICC_w"] = winsor(d.ICC_t2 - d.ICC_t1)
d["dICE_w"] = winsor(d.ICE_t2 - d.ICE_t1)
add("Dependent variable: current conditions (ICC)", d, y="dICC_w")
add("Dependent variable: expectations (ICE)", d, y="dICE_w")

# weights, in both the slopes and the interaction test
dw = d[d.WT_t1.notna() & (d.WT_t1 > 0)].copy()
add("Survey weight (WT) at the first interview", dw, w="WT_t1")

pd.DataFrame(rob).to_csv(f"{OUT}/table_robustness.csv", index=False)

# ------------------------------------------------------------------ additional checks (supplement)
extra = []


def add_extra(label, df, y="delta_sentiment_w", x="delta_px1_w", fe=None, cat_educ=False):
    df = df.copy()
    extra_cols = []
    if cat_educ:
        for k in range(2, 7):
            df[f"e{k}"] = (df.EDUC_t1 == k).astype(float)
            extra_cols.append(f"e{k}")
    if fe is not None:
        dums = pd.get_dummies(df[fe], prefix="fe", drop_first=True, dtype=float)
        df = pd.concat([df, dums], axis=1)
        extra_cols += list(dums.columns)
    base_ctrl = [c for c in BASE_X if not (cat_educ and c == "EDUC_t1")]
    # month fixed effects enter additively (common to both groups); slopes fully interacted
    X = pd.DataFrame({"_c": 1.0}, index=df.index)
    X["up"] = df.up
    X[x] = df[x]
    X[f"{x}__up"] = df[x] * df.up
    for c in base_ctrl:
        X[c] = df[c]
        X[f"{c}__up"] = df[c] * df.up
    for c in extra_cols:
        X[c] = df[c]
    rr = sm.OLS(df[y], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["person"])[0]})
    bd_, gm_ = rr.params[x], rr.params[f"{x}__up"]
    extra.append({"label": label, "b_down": float(bd_), "b_up": float(bd_ + gm_), "gamma": float(gm_),
                  "se_gamma": float(rr.bse[f"{x}__up"]), "p_gamma": float(rr.pvalues[f"{x}__up"]),
                  "lambda": float(abs(bd_ + gm_) / abs(bd_)), "n": int(rr.nobs)})


add_extra("Survey-month fixed effects (second interview)", d, fe="YYYYMM_t2")
add_extra("Survey-month fixed effects (first interview)", d, fe="YYYYMM_t1")
add_extra("Education as six categories", d, cat_educ=True)
add_extra("Excluding pairs involving the web mode", d[~d.web_pair])
# alternative thresholds: drop small changes (|dPX1| <= 1 or <= 2)
add_extra("Excluding changes of 1 point or less", d[d.delta_px1_w.abs() > 1])
add_extra("Excluding changes of 2 points or less", d[d.delta_px1_w.abs() > 2])
# un-winsorized
du = d.copy(); du["dics_raw"] = du.delta_sentiment; du["dpx1_raw"] = du.delta_px1
du = du[du.dpx1_raw != 0]; du["up"] = (du.dpx1_raw > 0).astype(int)
add_extra("No winsorization", du, y="dics_raw", x="dpx1_raw")
pd.DataFrame(extra).to_csv(f"{OUT}/table_additional_checks.csv", index=False)

# ------------------------------------------------------------------ descriptive statistics
desc = {}
for name, sub in [("full", d), ("increase", d[d.up == 1]), ("decrease", d[d.up == 0])]:
    desc[name] = {
        "n": int(len(sub)),
        "dICS_mean": float(sub.delta_sentiment_w.mean()), "dICS_sd": float(sub.delta_sentiment_w.std()),
        "dPX1_mean": float(sub.delta_px1_w.mean()), "dPX1_sd": float(sub.delta_px1_w.std()),
        "abs_dPX1_mean": float(sub.delta_px1_w.abs().mean()),
        "ICS_t1_mean": float(sub.ICS_t1.mean()), "ICS_t1_sd": float(sub.ICS_t1.std()),
        "PX1_t1_mean": float(sub.PX1_t1.mean()), "PX1_t1_sd": float(sub.PX1_t1.std()),
        "age_mean": float(sub.AGE_t1.mean()), "age_sd": float(sub.AGE_t1.std()),
        "income_mean": float(sub.INCOME_t1.mean()), "income_sd": float(sub.INCOME_t1.std()),
        "college_share": float((sub.EDUC_t1 >= 5).mean()),
        "region_shares": {k: float((sub.REGION_t1 == k).mean()) for k in (1, 2, 3, 4)},
    }
results["descriptives"] = desc

# ICS standard deviation at three levels
ics_ind = raw.ICS.dropna()
monthly = raw.dropna(subset=["ICS"]).groupby("YYYYMM").apply(
    lambda g: np.average(g.ICS, weights=g.WT.where(g.WT > 0).fillna(0)) if (g.WT > 0).any() else g.ICS.mean())
results["ics_sd_levels"] = {
    "individual_cross_section_sd_all_records": float(ics_ind.std()),
    "individual_mean_within_month_sd": float(raw.groupby("YYYYMM").ICS.std().mean()),
    "monthly_weighted_index_sd": float(monthly.std()),
    "within_person_change_sd_analytic_sample": float(d.delta_sentiment_w.std()),
}

# ------------------------------------------------------------------ monthly PX1 series, rebuilt
raw["PX1c"] = clean_px(raw.PX1)


def wmedian(v, w):
    o = np.argsort(v); v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


rowsm = []
for ym, g in raw.groupby("YYYYMM"):
    gg = g.dropna(subset=["PX1c"])
    w = gg.WT.where(gg.WT > 0).fillna(0).values
    if w.sum() == 0:
        w = np.ones(len(gg))
    rowsm.append({"YYYYMM": ym,
                  "ICS_weighted_mean": float(np.average(g.ICS.dropna(), weights=np.where(
                      g.dropna(subset=['ICS']).WT > 0, g.dropna(subset=['ICS']).WT, 0)) if (g.dropna(subset=['ICS']).WT > 0).any() else g.ICS.mean()),
                  "PX1_weighted_median": wmedian(gg.PX1c.values.astype(float), w),
                  "PX1_weighted_mean": float(np.average(gg.PX1c, weights=w)),
                  "PX1_unweighted_mean_old_with_missing_codes": float(g.PX1.mean()),
                  "PX1_share_missing": float(1 - len(gg) / len(g)),
                  "n": int(len(g)), "n_px1_valid": int(len(gg))})
ms = pd.DataFrame(rowsm)
ms.to_csv(f"{OUT}/monthly_series_rebuilt.csv", index=False)
results["monthly_px1_check"] = ms.set_index("YYYYMM").loc[
    [m for m in (198005, 200807, 202206, 202504, 202505) if m in set(ms.YYYYMM)],
    ["PX1_weighted_median", "PX1_weighted_mean", "PX1_unweighted_mean_old_with_missing_codes", "ICS_weighted_mean"]
].to_dict(orient="index")

# economic magnitude, at the change level
sd_change = results["ics_sd_levels"]["within_person_change_sd_analytic_sample"]
results["magnitude"] = {
    "ten_point_rise_effect": 10 * main["b_up"], "ten_point_fall_effect": 10 * main["b_down"],
    "ten_point_rise_in_sd_of_change": 10 * main["b_up"] / sd_change,
    "ten_point_fall_in_sd_of_change": 10 * main["b_down"] / sd_change,
    "sd_dPX1_analytic": float(d.delta_px1_w.std()),
    "one_sd_dPX1_rise_effect": float(d.delta_px1_w.std() * main["b_up"]),
}

main.pop("_res"); main_hc1.pop("_res")
results["log"] = log
with open(f"{OUT}/revision_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(json.dumps({k: v for k, v in results.items() if k != "descriptives"}, indent=1, default=float))
