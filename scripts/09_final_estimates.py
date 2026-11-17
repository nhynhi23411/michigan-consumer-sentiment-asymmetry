"""
Final estimates for the submission package (30 September 2026).

Every number in the Main Manuscript, the Supplementary Materials and the Full
Manuscript is produced here (or in 06-08 for diagnostics) from unrounded values.

Specification (one model delivers slopes, slope difference, lambda and its CI):
  dICS_i = a + b_down*dPX1_i + d*Up_i + g*(dPX1_i x Up_i)
           + X_i'th + (X_i x Up_i)'th_up [+ survey-month FE] + e_i
  b_up = b_down + g ;  lambda = |b_up| / |b_down|
  This is numerically identical to separate OLS regressions for the two groups.
Inference: SEs clustered by respondent (a respondent may contribute two consecutive
pairs). 95% CI for lambda: percentile respondent-cluster bootstrap, 2,000 draws,
seed 42, re-estimating the same specification in every draw.
Outputs: output/final/*.csv and output/final/final_results.json
"""
import json
import os
import platform
import warnings

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm

warnings.filterwarnings("ignore")
OUT = "output/final"
os.makedirs(OUT, exist_ok=True)
SEED, NBOOT = 42, 2000

p = pd.read_csv("data/processed/panel_clean.csv")
s2 = p.IDPREV_t1.notna()
p["person"] = (np.where(s2, p.IDPREV_t1, p.ID_t1).astype("int64").astype(str) + "_" +
               np.where(s2, p.DATEPR_t1, p.YYYYMM_t1).astype("int64").astype(str))
OK = p.EDUC_t1.between(1, 6) & p.REGION_t1.between(1, 4) & p.AGE_t1.between(18, 97) & (p.INCOME_t1 >= 0)
p["inc"] = p.INCOME_t1 / 1e4
for k in (2, 3, 4):
    p[f"r{k}"] = (p.REGION_t1 == k).astype(float)
p["web"] = ((p.METHOD_t1 == 2) & (p.METHOD_t2 == 2)).astype(float)
CTRL = ["AGE_t1", "inc", "EDUC_t1", "r2", "r3", "r4"]


def winsor(s):
    a, b = s.quantile([0.01, 0.99])
    return s.clip(a, b)


def clean_px(s):
    return s.where((s != -97) & (s < 96))


def make(df, xcol):
    d = df[(df[xcol] != 0)].copy()
    d["x"] = d[xcol]
    d["up"] = (d.x > 0).astype(float)
    return d


BASE = make(p[OK], "delta_px1_w")
BASE["y"] = BASE.delta_sentiment_w


def X_of(d, ctrl=CTRL):
    cols = {"c": np.ones(len(d)), "x": d.x.values}
    for c in ctrl:
        cols[c] = d[c].values
    X = pd.DataFrame(cols, index=d.index)
    for c in list(X.columns):
        X[c + "_u"] = X[c] * d.up.values
    return X


def estimate(d, label, w=None, fe=None, ctrl=CTRL, boot=True, seed=SEED):
    d = d.copy()
    X = X_of(d, ctrl)
    y = d.y.astype(float)
    if fe is not None:
        X = pd.concat([X, pd.get_dummies(d[fe], prefix="fe", drop_first=True, dtype=float)], axis=1)
    groups = pd.factorize(d.person)[0]
    if w is None:
        r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        r = sm.WLS(y, X, weights=d[w]).fit(cov_type="cluster", cov_kwds={"groups": groups})
    bd, g = r.params["x"], r.params["x_u"]
    V = r.cov_params()
    se_up = np.sqrt(V.loc["x", "x"] + V.loc["x_u", "x_u"] + 2 * V.loc["x", "x_u"])
    out = {"label": label, "pairs": int(len(d)), "n_up": int(d.up.sum()), "n_down": int(len(d) - d.up.sum()),
           "respondents": int(d.person.nunique()), "first_t1": int(d.YYYYMM_t1.min()), "last_t2": int(d.YYYYMM_t2.max()),
           "b_down": bd, "se_down": r.bse["x"], "b_up": bd + g, "se_up": se_up,
           "gamma": g, "se_gamma": r.bse["x_u"], "p_gamma": r.pvalues["x_u"],
           "lambda": abs(bd + g) / abs(bd), "r2": r.rsquared}
    grad = pd.Series(0.0, index=r.params.index)
    grad["x"], grad["x_u"] = -g / bd ** 2, 1 / bd
    se_l = float(np.sqrt(grad @ V @ grad))
    out["lambda_ci_delta_lo"], out["lambda_ci_delta_hi"] = out["lambda"] - 1.96 * se_l, out["lambda"] + 1.96 * se_l
    if boot:
        # Respondent-cluster bootstrap implemented with frequency weights: drawing G
        # respondents with replacement is equivalent to weighting every row of respondent
        # k by the number of times k is drawn.
        Xb = X_of(d, ctrl).values
        yb = y.values.astype(float)
        cols = list(X_of(d.head(2), ctrl).columns)
        jx, jg = cols.index("x"), cols.index("x_u")
        sw = d[w].values if w is not None else np.ones(len(d))
        fe_codes = pd.factorize(d[fe])[0] if fe is not None else None
        G = groups.max() + 1
        rng = np.random.default_rng(seed)
        lams, gams = np.empty(NBOOT), np.empty(NBOOT)
        for b in range(NBOOT):
            cnt = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
            wr = cnt[groups] * sw
            keep = wr > 0
            Xr, yr, wk = Xb[keep], yb[keep], wr[keep]
            if fe_codes is not None:
                f = fe_codes[keep]
                ws = np.bincount(f, wk)
                ws[ws == 0] = 1
                yr = yr - (np.bincount(f, wk * yr) / ws)[f]
                Xm = np.vstack([np.bincount(f, wk * Xr[:, j]) / ws for j in range(Xr.shape[1])]).T
                Xr = (Xr - Xm[f])[:, 1:]
                off = 1
            else:
                off = 0
            q = np.sqrt(wk)
            beta, *_ = np.linalg.lstsq(Xr * q[:, None], yr * q, rcond=None)
            bdb, gb = beta[jx - off], beta[jg - off]
            lams[b], gams[b] = abs(bdb + gb) / abs(bdb), gb
        out["lambda_ci_lo"], out["lambda_ci_hi"] = np.percentile(lams, [2.5, 97.5])
        out["lambda_boot_median"] = float(np.median(lams))
        out["boot_share_lambda_le_1"] = float((lams <= 1).mean())
        out["gamma_ci_lo"], out["gamma_ci_hi"] = np.percentile(gams, [2.5, 97.5])
        if label.startswith("Baseline"):
            pd.Series(lams, name="lambda").to_csv(f"{OUT}/bootstrap_lambda_baseline.csv", index=False)
    print(f"{label:60s} lam={out['lambda']:.3f} g={g:.3f} p={out['p_gamma']:.4f} N={len(d)}", flush=True)
    return out, r


R = {}
# ---------------- Table 1: main results
t1 = []
o, r_main = estimate(BASE, "Baseline, full sample"); t1.append(o)
o, _ = estimate(BASE, "Full sample, survey-month fixed effects", fe="YYYYMM_t2"); t1.append(o)
o, _ = estimate(BASE[BASE.web == 0], "Telephone pairs, 1980-2024"); t1.append(o)
o, _ = estimate(BASE[BASE.web == 1], "Web pairs, 2024-2026"); t1.append(o)
pd.DataFrame(t1).to_csv(f"{OUT}/table1_main.csv", index=False)

# formal test: gamma differs between web and telephone (all coefficients interacted with web)
d = BASE.copy()
X = X_of(d)
for c in list(X.columns):
    X[c + "_w"] = X[c] * d.web.values
rw = sm.OLS(d.y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.person)[0]})
R["gamma_web_minus_phone"] = {"coef": rw.params["x_u_w"], "se": rw.bse["x_u_w"], "p": rw.pvalues["x_u_w"]}

# full coefficient table (controls by group) for the supplement
rows = []
V = r_main.cov_params()
for c in ["c", "x"] + CTRL:
    se_u = np.sqrt(V.loc[c, c] + V.loc[c + "_u", c + "_u"] + 2 * V.loc[c, c + "_u"])
    rows.append({"term": c, "decrease_b": r_main.params[c], "decrease_se": r_main.bse[c],
                 "increase_b": r_main.params[c] + r_main.params[c + "_u"], "increase_se": se_u,
                 "difference_b": r_main.params[c + "_u"], "difference_se": r_main.bse[c + "_u"],
                 "difference_p": r_main.pvalues[c + "_u"]})
pd.DataFrame(rows).to_csv(f"{OUT}/tableS_main_full_coefficients.csv", index=False)

# restricted pooled model (common control coefficients), the old Table 3 column 3
Xr = pd.DataFrame({"c": 1.0, "x": d.x, "up": d.up, "x_u": d.x * d.up}, index=d.index)
for c in CTRL:
    Xr[c] = d[c]
rr = sm.OLS(d.y, Xr).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.person)[0]})
bd, g = rr.params["x"], rr.params["x_u"]
R["restricted_pooled"] = {"b_down": bd, "se_down": rr.bse["x"], "gamma": g, "se_gamma": rr.bse["x_u"],
                          "p_gamma": rr.pvalues["x_u"], "b_up": bd + g, "lambda": abs(bd + g) / abs(bd),
                          "up_indicator": rr.params["up"], "se_up_indicator": rr.bse["up"], "r2": rr.rsquared}
R["hc1_main"] = {k: float(v) for k, v in zip(
    ["se_down", "se_gamma"],
    sm.OLS(BASE.y, X_of(BASE)).fit(cov_type="HC1").bse[["x", "x_u"]])}

# ---------------- Table 2: baseline + seven checks
t2 = [t1[0] | {"label": "Baseline (Table 1, column 1)"}]
o, _ = estimate(BASE[BASE.METHOD_t1 == BASE.METHOD_t2], "Excluding pairs with a change in telephone mode"); t2.append(o)
o, _ = estimate(BASE[(BASE.YYYYMM_t1 < 202001) | (BASE.YYYYMM_t1 > 202212)], "Excluding first interviews in 2020-2022"); t2.append(o)
Z = p[OK].copy(); Z["x"] = Z.delta_px1_w; Z["up"] = (Z.x > 0).astype(float); Z["y"] = Z.delta_sentiment_w
o, _ = estimate(Z, "Zero-change pairs added to the non-increase group"); t2.append(o)
Q = p[OK].copy()
Q["dpx5"] = clean_px(Q.PX5_t2) - clean_px(Q.PX5_t1)
Q = Q.dropna(subset=["dpx5"]); Q["dpx5_w"] = winsor(Q.dpx5)
Q = make(Q, "dpx5_w"); Q["y"] = Q.delta_sentiment_w
o, _ = estimate(Q, "Five-to-ten-year expectations (PX5)"); t2.append(o)
B2 = BASE.copy(); B2["y"] = winsor(B2.ICC_t2 - B2.ICC_t1)
o, _ = estimate(B2, "Dependent variable: current conditions (ICC)"); t2.append(o)
B3 = BASE.copy(); B3["y"] = winsor(B3.ICE_t2 - B3.ICE_t1)
o, _ = estimate(B3, "Dependent variable: expectations (ICE)"); t2.append(o)
BW = BASE[BASE.WT_t1.notna() & (BASE.WT_t1 > 0)]
o, _ = estimate(BW, "Survey weight at the first interview (WT)", w="WT_t1"); t2.append(o)
pd.DataFrame(t2).to_csv(f"{OUT}/table2_robustness.csv", index=False)

# ---------------- Supplement: where the asymmetry comes from, and other checks
s = []
ph = BASE[BASE.web == 0]
for lo, hi, lab in [(198000, 198912, "1980s"), (199000, 199912, "1990s"), (200000, 200912, "2000s"),
                    (201000, 201912, "2010s"), (202000, 202312, "2020-2023")]:
    o, _ = estimate(ph[(ph.YYYYMM_t1 >= lo) & (ph.YYYYMM_t1 <= hi)], f"Telephone, first interview in {lab}"); s.append(o)
wb = BASE[BASE.web == 1]
o, _ = estimate(wb, "Web pairs, survey-month fixed effects", fe="YYYYMM_t2"); s.append(o)
o, _ = estimate(ph, "Telephone pairs, survey-month fixed effects", fe="YYYYMM_t2"); s.append(o)
o, _ = estimate(wb[wb.x.abs() <= 10], "Web pairs, changes of 10 points or less"); s.append(o)
o, _ = estimate(wb[wb.x.abs() > 10], "Web pairs, changes above 10 points"); s.append(o)
o, _ = estimate(ph[ph.x.abs() <= 10], "Telephone pairs, changes of 10 points or less"); s.append(o)
o, _ = estimate(ph[ph.x.abs() > 10], "Telephone pairs, changes above 10 points"); s.append(o)
lo_c, hi_c = BASE.x.min(), BASE.x.max()
o, _ = estimate(wb[(wb.x > lo_c) & (wb.x < hi_c)], "Web pairs, excluding changes at the winsorization limits"); s.append(o)
pd.DataFrame(s).to_csv(f"{OUT}/tableS_sources.csv", index=False)

a = []
o, _ = estimate(BASE, "Survey-month fixed effects, first interview", fe="YYYYMM_t1"); a.append(o)
E = BASE.copy()
for k in range(2, 7):
    E[f"e{k}"] = (E.EDUC_t1 == k).astype(float)
o, _ = estimate(E, "Education as five indicators", ctrl=["AGE_t1", "inc", "r2", "r3", "r4"] + [f"e{k}" for k in range(2, 7)]); a.append(o)
U = p[OK].copy(); U["x"] = U.delta_px1; U = U[U.x != 0]; U["up"] = (U.x > 0).astype(float); U["y"] = U.delta_sentiment
o, _ = estimate(U, "No winsorization"); a.append(o)
o, _ = estimate(BASE[BASE.x.abs() > 1], "Excluding changes of 1 point or less"); a.append(o)
o, _ = estimate(BASE[BASE.x.abs() > 2], "Excluding changes of 2 points or less"); a.append(o)
pd.DataFrame(a).to_csv(f"{OUT}/tableS_additional.csv", index=False)

# curvature versus kink (full sample): squared terms in each group
Xq = X_of(BASE)
Xq["x2"] = BASE.x ** 2
Xq["x2_u"] = Xq.x2 * BASE.up
rq = sm.OLS(BASE.y, Xq).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(BASE.person)[0]})
R["curvature"] = {"gamma_with_squares": rq.params["x_u"], "se": rq.bse["x_u"], "p": rq.pvalues["x_u"],
                  "x2_down": rq.params["x2"], "p_x2_down": rq.pvalues["x2"],
                  "x2_diff": rq.params["x2_u"], "p_x2_diff": rq.pvalues["x2_u"]}

# ---------------- descriptives
desc = []
for lab, g_ in [("Full sample", BASE), ("Increase group", BASE[BASE.up == 1]), ("Decrease group", BASE[BASE.up == 0]),
                ("Telephone pairs", ph), ("Web pairs", wb)]:
    desc.append({"sample": lab, "pairs": len(g_),
                 "dICS_mean": g_.y.mean(), "dICS_sd": g_.y.std(), "dPX1_mean": g_.x.mean(), "dPX1_sd": g_.x.std(),
                 "abs_dPX1_mean": g_.x.abs().mean(), "share_abs_dPX1_gt10": (g_.x.abs() > 10).mean(),
                 "ICS1_mean": g_.ICS_t1.mean(), "ICS1_sd": g_.ICS_t1.std(),
                 "PX1_1_mean": clean_px(g_.PX1_t1).mean(), "PX1_1_sd": clean_px(g_.PX1_t1).std(),
                 "age_mean": g_.AGE_t1.mean(), "age_sd": g_.AGE_t1.std(),
                 "income_mean": g_.INCOME_t1.mean(), "income_sd": g_.INCOME_t1.std(),
                 "college_share": (g_.EDUC_t1 >= 5).mean(),
                 "west": (g_.REGION_t1 == 1).mean(), "north_central": (g_.REGION_t1 == 2).mean(),
                 "northeast": (g_.REGION_t1 == 3).mean(), "south": (g_.REGION_t1 == 4).mean()})
pd.DataFrame(desc).to_csv(f"{OUT}/tableS_descriptives.csv", index=False)

# ---------------- sample flow
raw_n = 342345
R["flow"] = {
    "raw_records": raw_n,
    "reinterview_records": 124705,
    "linked_pairs": 124704,
    "pairs_valid_dICS_dPX1": int(len(p)),
    "zero_change": int((p.delta_px1_w == 0).sum()),
    "nonzero": int((p.delta_px1_w != 0).sum()),
    "missing_controls_among_nonzero": int(((p.delta_px1_w != 0) & ~OK).sum()),
    "analytic_pairs": int(len(BASE)),
    "respondents": int(BASE.person.nunique()),
    "respondents_two_pairs": int((BASE.person.value_counts() == 2).sum()),
    "telephone_pairs": int((BASE.web == 0).sum()), "web_pairs": int((BASE.web == 1).sum()),
    "telephone_mode_change_pairs": int((BASE.METHOD_t1 != BASE.METHOD_t2).sum()),
    "phone_web_pairs": int(((BASE.METHOD_t1 == 2) != (BASE.METHOD_t2 == 2)).sum()),
    "winsor_limits_dPX1": [float(BASE.x.min()), float(BASE.x.max())],
    "winsor_limits_dICS": [float(BASE.y.min()), float(BASE.y.max())],
    "first_t1": int(BASE.YYYYMM_t1.min()), "last_t1": int(BASE.YYYYMM_t1.max()),
    "first_t2": int(BASE.YYYYMM_t2.min()), "last_t2": int(BASE.YYYYMM_t2.max()),
    "web_first_t1": int(wb.YYYYMM_t1.min()), "phone_last_t2": int(ph.YYYYMM_t2.max()),
}
R["sd_levels"] = {"dICS_within_person_sd": float(BASE.y.std()),
                  "ICS_individual_sd_first_interview": float(BASE.ICS_t1.std())}
R["software"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                 "statsmodels": statsmodels.__version__, "seed": SEED, "bootstrap_draws": NBOOT,
                 "data_extract_date": "2026-09-05 (SDA archive, sda.umsurvey.org)"}
json.dump(R, open(f"{OUT}/final_results.json", "w"), indent=2, default=float)
print(json.dumps(R, indent=1, default=float))
