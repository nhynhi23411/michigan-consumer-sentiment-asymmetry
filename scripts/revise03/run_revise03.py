"""Checks added in response to the referee-style review (level scale, no bootstrap except where stated).
 A  continuity-constrained kink (common intercept: delta = 0) and continuous quadratic spline
 B  heaping of expected inflation at multiples of 5 and 10
 C  symmetric band around zero (telephone / web separately)
 D  inflation regime: telephone pairs with first interview in 2021-2023 versus earlier
 E  heterogeneity by age, income, education and initial ICS
usage (from repo root): python scripts/revise03/run_revise03.py ; env OUT (default output/revise03)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revise01"))
from core import *

OUT = os.environ.get("OUT", "output/revise03")
os.makedirs(OUT, exist_ok=True)
D = load("level")
rows = []


def add(block, label, s, kind="base", drop_delta=False):
    s = s.reset_index(drop=True)
    X = design(s, kind)
    if drop_delta:
        X = X.drop(columns=["c_u"])
    g = pd.factorize(s.person)[0]
    r = sm.OLS(s.y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    bd, gm = r.params["x"], r.params["x_u"]
    rows.append({"block": block, "spec": label, "n": len(s), "n_up": int(s.up.sum()), "fall_slope": bd, "rise_slope": bd + gm,
                 "gamma": gm, "se_gamma": r.bse["x_u"], "p_gamma": r.pvalues["x_u"], "lambda": abs(bd + gm) / abs(bd),
                 "delta": r.params.get("c_u", np.nan), "p_delta": r.pvalues.get("c_u", np.nan)})
    print(f"{block:12s} {label:55s} n={len(s):6d} lam={abs(bd + gm) / abs(bd):6.3f} gamma={gm:7.3f} p={r.pvalues['x_u']:.4f}", flush=True)


samples = {"Pooled": D, "Telephone": D[D.web == 0], "Web": D[D.web == 1]}
# A: continuity
for nm, s in samples.items():
    add("A_kink", f"{nm}: baseline (jump delta free)", s)
    add("A_kink", f"{nm}: continuous kink (delta = 0)", s, drop_delta=True)
    add("A_kink", f"{nm}: continuous kink + quadratic (delta = 0)", s, kind="quad", drop_delta=True)
# B: heaping
for nm, s in samples.items():
    m5 = (s.PX1_t1 % 5 == 0) | (s.PX1_t2 % 5 == 0)
    m10 = (s.PX1_t1 % 10 == 0) | (s.PX1_t2 % 10 == 0)
    add("B_heaping", f"{nm}: all pairs", s)
    add("B_heaping", f"{nm}: excluding pairs with a reported value at a multiple of 5", s[~m5])
    add("B_heaping", f"{nm}: excluding pairs with a reported value at a multiple of 10", s[~m10])
# C: symmetric band
for nm, s in samples.items():
    for c in (1, 2):
        add("C_band", f"{nm}: excluding |dPX1| <= {c}", s[s.x.abs() > c])
# D: regime
tel = D[D.web == 0]
surge = (tel.YYYYMM_t1 >= 202101) & (tel.YYYYMM_t1 <= 202312)
add("D_regime", "Telephone, first interview 2021-2023", tel[surge])
add("D_regime", "Telephone, first interview before 2021", tel[tel.YYYYMM_t1 < 202101])
add("D_regime", "Web (2024-2026)", D[D.web == 1])
# E: heterogeneity (pooled sample)
q = lambda s, col, k: pd.qcut(s[col], k, labels=False, duplicates="drop")
D["agegrp"] = pd.cut(D.AGE_t1, [0, 34, 54, 120], labels=["18-34", "35-54", "55+"])
D["incq"] = q(D, "INCOME_t1", 3)
D["icsq"] = q(D, "ICS_t1", 3)
for lab, mask in [("age 18-34", D.agegrp == "18-34"), ("age 35-54", D.agegrp == "35-54"), ("age 55+", D.agegrp == "55+"),
                  ("income tercile 1 (low)", D.incq == 0), ("income tercile 2", D.incq == 1), ("income tercile 3 (high)", D.incq == 2),
                  ("no college (EDUC <= 4)", D.EDUC_t1 <= 4), ("college (EDUC >= 5)", D.EDUC_t1 >= 5),
                  ("initial ICS tercile 1 (low)", D.icsq == 0), ("initial ICS tercile 2", D.icsq == 1), ("initial ICS tercile 3 (high)", D.icsq == 2)]:
    add("E_heterog", "Pooled, " + lab, D[mask])
pd.DataFrame(rows).to_csv(f"{OUT}/review_checks.csv", index=False)
