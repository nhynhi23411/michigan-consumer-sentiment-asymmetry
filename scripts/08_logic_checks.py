"""
Checks for the revised framing (Sept 2026):
 1. Formal test that the slope asymmetry (gamma) differs between web and telephone pairs.
 2. Web result without pairs at the winsorization caps (noise / extreme answers).
 3. Lambda CI (delta method) in telephone and web samples.
 4. Kink versus curvature: add squared change terms in each group.
Respondent-clustered SEs throughout. Output: output/revision/logic_checks.json
"""
import json, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
p = pd.read_csv("data/processed/panel_clean.csv")
s2 = p.IDPREV_t1.notna()
p["person"] = (np.where(s2, p.IDPREV_t1, p.ID_t1).astype("int64").astype(str) + "_" +
               np.where(s2, p.DATEPR_t1, p.YYYYMM_t1).astype("int64").astype(str))
ok = p.EDUC_t1.between(1, 6) & p.REGION_t1.between(1, 4) & p.AGE_t1.between(18, 97) & (p.INCOME_t1 >= 0)
d = p[(p.delta_px1_w != 0) & ok].copy()
d["up"] = (d.delta_px1_w > 0).astype(float)
d["inc"] = d.INCOME_t1 / 1e4
for k in (2, 3, 4):
    d[f"r{k}"] = (d.REGION_t1 == k).astype(float)
d["web"] = ((d.METHOD_t1 == 2) & (d.METHOD_t2 == 2)).astype(float)
C = ["AGE_t1", "inc", "EDUC_t1", "r2", "r3", "r4"]


def fit(df, extra=None, inter_web=False, sq=False):
    X = pd.DataFrame({"c": 1.0, "x": df.delta_px1_w}, index=df.index)
    for c in C:
        X[c] = df[c]
    if sq:
        X["x2"] = df.delta_px1_w ** 2
    for c in list(X.columns):
        X[c + "_u"] = X[c] * df.up
    if inter_web:
        for c in list(X.columns):
            X[c + "_w"] = X[c] * df.web
    return sm.OLS(df.delta_sentiment_w, X).fit(cov_type="cluster",
                                                cov_kwds={"groups": pd.factorize(df.person)[0]})


def lam(r, pre=""):
    bd, g = r.params[pre + "x"], r.params[pre + "x_u"]
    V = r.cov_params()
    grad = pd.Series(0.0, index=r.params.index)
    grad[pre + "x"], grad[pre + "x_u"] = -g / bd ** 2, 1 / bd
    se = float(np.sqrt(grad @ V @ grad))
    L = abs(bd + g) / abs(bd)
    return {"b_down": bd, "b_up": bd + g, "gamma": g, "se_gamma": r.bse[pre + "x_u"],
            "p_gamma": r.pvalues[pre + "x_u"], "lambda": L, "ci": [L - 1.96 * se, L + 1.96 * se], "n": int(r.nobs)}


out = {}
r = fit(d, inter_web=True)
out["gamma_difference_web_minus_phone"] = {"coef": r.params["x_u_w"], "se": r.bse["x_u_w"], "p": r.pvalues["x_u_w"]}
out["phone"] = lam(fit(d[d.web == 0]))
out["web"] = lam(fit(d[d.web == 1]))
lo, hi = d.delta_px1_w.min(), d.delta_px1_w.max()
w_in = d[(d.web == 1) & (d.delta_px1_w > lo) & (d.delta_px1_w < hi)]
out["web_excluding_winsor_caps"] = lam(fit(w_in))
out["winsor_caps"] = [lo, hi]
out["web_share_at_caps"] = float(((d[d.web == 1].delta_px1_w == lo) | (d[d.web == 1].delta_px1_w == hi)).mean())
out["phone_share_at_caps"] = float(((d[d.web == 0].delta_px1_w == lo) | (d[d.web == 0].delta_px1_w == hi)).mean())
w_small = d[(d.web == 1) & (d.delta_px1_w.abs() <= 10)]
out["web_abs_change_le_10"] = lam(fit(w_small))
rq = fit(d, sq=True)
out["curvature_full"] = {"x2_down": rq.params["x2"], "p_x2_down": rq.pvalues["x2"],
                         "x2_up_minus_down": rq.params["x2_u"], "p": rq.pvalues["x2_u"],
                         "kink_gamma_with_curvature": rq.params["x_u"], "p_kink": rq.pvalues["x_u"]}
json.dump(out, open("output/revision/logic_checks.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=1, default=float))
