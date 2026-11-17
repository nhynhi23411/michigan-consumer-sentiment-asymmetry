"""
Core functions for the replication of the referee-response checks (Revise_01_DHQuan).
Independent re-implementation: sample construction follows scripts/01_clean_panel.py and
scripts/09_final_estimates.py of the EL70134 package; models are written from the description.

Set environment variable PANEL to the path of data/processed/panel_clean.csv
(output of 01_clean_panel.py).
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

PANEL = os.environ.get("PANEL", "data/processed/panel_clean.csv")
CTRL = ["AGE_t1", "inc", "EDUC_t1", "r2", "r3", "r4"]


def winsor(s):
    a, b = s.quantile([0.01, 0.99])
    return s.clip(a, b)


def load(scale="level"):
    """scale = 'level' (percentage-point change) or 'log' (100*dlog(1+PX1/100), the 'log-return' change)."""
    p = pd.read_csv(PANEL)
    s2 = p.IDPREV_t1.notna()
    p["person"] = (np.where(s2, p.IDPREV_t1, p.ID_t1).astype("int64").astype(str) + "_" +
                   np.where(s2, p.DATEPR_t1, p.YYYYMM_t1).astype("int64").astype(str))
    if scale == "log":
        # symmetric measure of the revision: ratio of gross expected inflation rates
        p["xraw"] = 100 * (np.log1p(p.PX1_t2 / 100) - np.log1p(p.PX1_t1 / 100))
        p["p0"] = 100 * np.log1p(p.PX1_t1 / 100)
        p["x"] = winsor(p.xraw)
    else:
        p["x"] = p.delta_px1_w
        p["p0"] = p.PX1_t1
    ok = (p.EDUC_t1.between(1, 6) & p.REGION_t1.between(1, 4) & p.AGE_t1.between(18, 97)
          & (p.INCOME_t1 >= 0) & (p.x != 0))
    d = p[ok].copy()
    d["y"] = d.delta_sentiment_w
    d["up"] = (d.x > 0).astype(float)
    d["inc"] = d.INCOME_t1 / 1e4
    for k in (2, 3, 4):
        d[f"r{k}"] = (d.REGION_t1 == k).astype(float)
    d["web"] = ((d.METHOD_t1 == 2) & (d.METHOD_t2 == 2)).astype(float)
    return d.reset_index(drop=True)


def design(d, kind="base"):
    """kind: base | quad | init | quad_init. Every term is fully interacted with Up (group-specific)."""
    up = d.up.values
    cols = {"c": np.ones(len(d)), "x": d.x.values}
    for c in CTRL:
        cols[c] = d[c].values
    if kind in ("quad", "quad_init"):
        cols["x2"] = d.x.values ** 2
    if kind in ("init", "quad_init"):
        cols["p0"] = d.p0.values
        cols["xp0"] = d.x.values * d.p0.values
    X = pd.DataFrame(cols, index=d.index)
    for c in list(X.columns):
        X[c + "_u"] = X[c] * up
    return X


def fit(d, kind="base", w=None):
    X = design(d, kind)
    g = pd.factorize(d.person)[0]
    if w is None:
        r = sm.OLS(d.y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    else:
        r = sm.WLS(d.y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": g})
    bd, gam = r.params["x"], r.params["x_u"]
    return r, {"n": len(d), "b_down": bd, "b_up": bd + gam, "gamma": gam, "se_gamma": r.bse["x_u"],
               "p_gamma": r.pvalues["x_u"], "lambda": abs(bd + gam) / abs(bd)}


def wald_p(r, terms):
    names = list(r.params.index)
    R = np.zeros((len(terms), len(names)))
    for j, t in enumerate(terms):
        R[j, names.index(t)] = 1
    return float(r.wald_test(R, scalar=True).pvalue)


def mode_diff(d, w=None):
    """Fully interacted with web: web-minus-telephone difference in the asymmetry coefficient gamma."""
    X = design(d, "base")
    XX = pd.concat([X, X.mul(d.web.values, axis=0).add_suffix("_w")], axis=1)
    g = pd.factorize(d.person)[0]
    if w is None:
        r = sm.OLS(d.y, XX).fit(cov_type="cluster", cov_kwds={"groups": g})
    else:
        r = sm.WLS(d.y, XX, weights=w).fit(cov_type="cluster", cov_kwds={"groups": g})
    return {"n": len(d), "n_web": int(d.web.sum()), "n_tel": int((1 - d.web).sum()),
            "diff": r.params["x_u_w"], "se": r.bse["x_u_w"], "p": r.pvalues["x_u_w"]}


def support_mask(d, mult=None):
    """1st-99th percentile intersection of telephone and web, separately for x<0, x>0 and initial PX1."""
    m = np.ones(len(d), bool)
    mult = np.ones(len(d)) if mult is None else mult
    tel, web = (d.web.values == 0), (d.web.values == 1)
    thr = {}
    for up in (0, 1):
        idx = d.up.values == up
        lo, hi = -np.inf, np.inf
        for grp in (tel, web):
            v = wq(d.x.values[idx & grp], mult[idx & grp], [0.01, 0.99])
            lo, hi = max(lo, v[0]), min(hi, v[1])
        thr["up" if up else "down"] = (lo, hi)
        m &= np.where(idx, (d.x.values >= lo) & (d.x.values <= hi), True)
    lo, hi = -np.inf, np.inf
    for grp in (tel, web):
        v = wq(d.p0.values[grp], mult[grp], [0.01, 0.99])
        lo, hi = max(lo, v[0]), min(hi, v[1])
    thr["p0"] = (lo, hi)
    m &= (d.p0.values >= lo) & (d.p0.values <= hi)
    return m, thr


def wq(v, w, qs):
    """weighted quantile (pandas 'linear' convention when weights are all 1)."""
    if np.all(w == 1):
        return np.quantile(v, qs)
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, c, v)


def propensity(d, mult=None):
    """Overlap weights from a logit of web on observables (unpenalised IRLS, standardised regressors)."""
    A = pd.DataFrame({"x": d.x, "p0": d.p0, "age": d.AGE_t1, "inc": d.inc, "edu": d.EDUC_t1,
                      "r2": d.r2, "r3": d.r3, "r4": d.r4, "up": d.up, "absx": d.x.abs(), "x2": d.x ** 2})
    Z = (A - A.mean()) / A.std()
    Z.insert(0, "c", 1.0)
    fw = np.ones(len(d)) if mult is None else mult
    r = sm.GLM(d.web.values, Z.values, family=sm.families.Binomial(), freq_weights=fw).fit()
    e = np.clip(r.predict(Z.values), 0.01, 0.99)
    return np.where(d.web.values == 1, 1 - e, e), Z
