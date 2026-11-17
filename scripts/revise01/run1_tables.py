"""Checks 1-3 for the level scale and the log scale:
 (1) restriction by size of the revision (web), (2) quadratic / initial-expectation specifications,
 (3) common support + overlap weights (point estimates)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from core import *

OUT = os.environ.get("OUT", "output/revise01")
os.makedirs(OUT, exist_ok=True)
scale = sys.argv[1] if len(sys.argv) > 1 else "level"
d = load(scale)
print(scale, "N", len(d), "web", int(d.web.sum()), flush=True)

# (1) magnitude restriction on the web sample
rows = []
wb = d[d.web == 1]
for c in (5, 10, 15, 20, None):
    s = wb if c is None else wb[wb.x.abs() <= c]
    _, o = fit(s)
    rows.append({"sample": "Toàn bộ web" if c is None else f"|dx| <= {c}", **o})
pd.DataFrame(rows).to_csv(f"{OUT}/t1_magnitude_{scale}.csv", index=False, encoding="utf-8")
print(pd.DataFrame(rows).round(4).to_string(), flush=True)

# (2) four specifications x three samples
rows = []
for name, s in (("Pooled", d), ("Tele", d[d.web == 0]), ("Web", wb)):
    for kind in ("base", "quad", "init", "quad_init"):
        r, o = fit(s, kind)
        o.update(sample=name, model=kind)
        if "quad" in kind:
            o["p_quad_joint"] = wald_p(r, ["x2", "x2_u"])
        if "init" in kind:
            o["p_init_joint"] = wald_p(r, ["xp0", "xp0_u"])
        rows.append(o)
t2 = pd.DataFrame(rows)
t2.to_csv(f"{OUT}/t2_nonlinear_{scale}.csv", index=False, encoding="utf-8")
print(t2[["sample", "model", "lambda", "p_gamma", "p_quad_joint", "p_init_joint"]].round(4).to_string(), flush=True)

# (3) common support and overlap weights
m, thr = support_mask(d)
s = d[m].reset_index(drop=True)
w, Z = propensity(s)
res = [{"spec": "Mau ban dau", **mode_diff(d)},
       {"spec": "Common support", **mode_diff(s)},
       {"spec": "Common support + overlap weighting", **mode_diff(s, w)}]
t3 = pd.DataFrame(res)
t3.to_csv(f"{OUT}/t3_mode_{scale}.csv", index=False, encoding="utf-8")
print(thr)
print(t3.round(4).to_string(), flush=True)
# balance (standardised mean differences) and ESS
bal = []
a = s.web.values == 0
for col in Z.columns[1:]:
    z = Z[col].values

    def smd(ww):
        m0, m1 = np.average(z[a], weights=ww[a]), np.average(z[~a], weights=ww[~a])
        v0 = np.average((z[a] - m0) ** 2, weights=ww[a])
        v1 = np.average((z[~a] - m1) ** 2, weights=ww[~a])
        return (m1 - m0) / np.sqrt((v0 + v1) / 2)
    bal.append({"var": col, "smd_before": smd(np.ones(len(s))), "smd_after": smd(w)})
pd.DataFrame(bal).to_csv(f"{OUT}/t3_balance_{scale}.csv", index=False, encoding="utf-8")
print(pd.DataFrame(bal).round(3).to_string())
print("ESS tel", w[a].sum() ** 2 / (w[a] ** 2).sum(), "web", w[~a].sum() ** 2 / (w[~a] ** 2).sum())
