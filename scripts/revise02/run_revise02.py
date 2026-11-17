"""S11 of the Supplementary Materials (level scale).
 S11.1 inference with clustering by survey month (Table S10)
 S11.2 controls for initial ICS / PX1 and intercept difference delta (Table S11)
 S11.3 marginal slopes by mode, common support + overlap weights, full-pipeline bootstrap (Table S12)
usage: python scripts/revise02/run_revise02.py [part] [NBOOT]    part in {1,2,3,all}; seed 42; env NPROC, OUT
Run from the repository root; needs data/processed/panel_clean.csv."""
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "revise01"))
from core import *

OUT = os.environ.get("OUT", "output/revise02")
os.makedirs(OUT, exist_ok=True)
PART = sys.argv[1] if len(sys.argv) > 1 else "all"
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SEED = 42
D = load("level")
D["month"] = D.YYYYMM_t2
PID, PINV = np.unique(D.person.values, return_inverse=True)


def centred(d):
    d = d.copy()
    for c in CTRL:
        d[c] = d[c] - d[c].mean()
    return d


def wls(X, y, w):
    return np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y * w))


# ---------------------------------------------------------------- S11.1
def month_boot(seed, X, y, mon_inv, fe):
    """Resample survey months; with fe=True each drawn month keeps its own fixed effect."""
    rng = np.random.default_rng(seed)
    nm = mon_inv.max() + 1
    w = np.bincount(rng.integers(0, nm, nm), minlength=nm)[mon_inv].astype(float)
    k = w > 0
    Xk, yk, wk = X[k], y[k], w[k]
    if fe:
        f = mon_inv[k]
        ws = np.maximum(np.bincount(f, wk), 1e-12)
        yk = yk - (np.bincount(f, wk * yk) / ws)[f]
        Xm = np.vstack([np.bincount(f, wk * Xk[:, j]) / ws for j in range(Xk.shape[1])]).T
        Xk = (Xk - Xm[f])[:, 1:]
        b = wls(Xk, yk, wk)
        return np.concatenate([[0.0], b])
    return wls(Xk, yk, wk)


def part1(pool):
    rows = []
    for lab, s, fe in (("Full sample", D, False), ("Full sample, month fixed effects", D, True),
                       ("Telephone pairs", D[D.web == 0], False), ("Web pairs", D[D.web == 1], False)):
        s = s.reset_index(drop=True)
        X = design(s, "base")
        cols = list(X.columns)
        Xf = pd.concat([X, pd.get_dummies(s.month, prefix="fe", drop_first=True, dtype=float)], axis=1) if fe else X
        g_p, g_m = pd.factorize(s.person)[0], pd.factorize(s.month)[0]
        res = {}
        for name, grp in (("Respondent", g_p), ("Survey month", g_m), ("Two-way", np.column_stack([g_p, g_m]))):
            r = sm.OLS(s.y, Xf).fit(cov_type="cluster", cov_kwds={"groups": grp})
            res[name] = (r.params["x_u"], r.bse["x_u"], r.pvalues["x_u"])
        seeds = np.random.SeedSequence(SEED).generate_state(NB)
        bs = pool.starmap(month_boot, [(sd, X.values, s.y.values, g_m, fe) for sd in seeds], chunksize=20)
        jx, ju = cols.index("x"), cols.index("x_u")
        L = np.array([abs(b[jx] + b[ju]) / abs(b[jx]) for b in bs])
        lo, hi = np.percentile(L, [2.5, 97.5])
        for name, (g, se, p) in res.items():
            rows.append({"sample": lab, "clustering": name, "gamma": g, "se": se, "p": p,
                         "lam_lo_month_boot": lo, "lam_hi_month_boot": hi})
        print(lab, {k: np.round(v, 4) for k, v in res.items()}, "lambda month-bootstrap", round(lo, 3), round(hi, 3), flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/tableS10_month_clustering.csv", index=False)


# ---------------------------------------------------------------- S11.2
def part2():
    rows = []
    for lab, s in (("Full sample", D), ("Telephone", D[D.web == 0]), ("Web", D[D.web == 1])):
        s = centred(s).reset_index(drop=True)
        s["ics0"] = s.ICS_t1 - s.ICS_t1.mean()
        s["px0"] = s.PX1_t1 - s.PX1_t1.mean()
        for spec, extra in (("Baseline, centered", []), ("+ initial ICS", ["ics0"]),
                            ("+ initial ICS and PX1", ["ics0", "px0"])):
            X = design(s, "base")
            for c in extra:
                X[c] = s[c]
                X[c + "_u"] = s[c] * s.up
            r = sm.OLS(s.y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.person)[0]})
            V = r.cov_params()
            se_up = float(np.sqrt(V.loc["x", "x"] + V.loc["x_u", "x_u"] + 2 * V.loc["x", "x_u"]))
            rows.append({"sample": lab, "spec": spec, "fall_slope": r.params["x"], "fall_se": r.bse["x"],
                         "rise_slope": r.params["x"] + r.params["x_u"], "rise_se": se_up,
                         "gamma": r.params["x_u"], "gamma_se": r.bse["x_u"], "p_gamma": r.pvalues["x_u"],
                         "delta": r.params["c_u"], "delta_se": r.bse["c_u"], "n": len(s)})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/tableS11_initial_levels.csv", index=False)
    print(t.round(3).to_string(), flush=True)
    print("mean initial ICS, increase vs decrease:", round(D[D.up == 1].ICS_t1.mean(), 2), round(D[D.up == 0].ICS_t1.mean(), 2),
          "| mean initial PX1:", round(D[D.up == 1].PX1_t1.mean(), 2), round(D[D.up == 0].PX1_t1.mean(), 2), flush=True)


# ---------------------------------------------------------------- S11.3
SC = [(p0, a) for p0 in (3, 4, 6) for a in (2, 5)]


def slopes_diff(b, cols):
    g = lambda n: b[cols.index(n)]
    out = []
    for p0, a in SC:
        sd = g("x") + 2 * g("x2") * (-a) + g("xp0") * p0
        su = g("x") + g("x_u") + 2 * (g("x2") + g("x2_u")) * a + (g("xp0") + g("xp0_u")) * p0
        out.append(su - sd)
    return np.array(out)


def pipeline(d, mult):
    """support -> logit -> overlap weights -> separate quad+initial models by mode -> rise-minus-fall differences."""
    m, _ = support_mask(d, mult)
    s, mu = d[m].reset_index(drop=True), mult[m]
    w, _ = propensity(s, mu)
    res = []
    for web in (0, 1):
        k = s.web.values == web
        X = design(s[k], "quad_init")
        b = wls(X.values, s.y.values[k], (w * mu)[k])
        res.append(slopes_diff(b, list(X.columns)))
    return np.concatenate([res[0], res[1], res[1] - res[0]])


def draw3(seed):
    rng = np.random.default_rng(seed)
    cnt = np.bincount(rng.integers(0, len(PID), len(PID)), minlength=len(PID))
    mult = cnt[PINV].astype(float)
    k = mult > 0
    return pipeline(D[k].reset_index(drop=True), mult[k])


def part3(pool):
    pt = pipeline(D, np.ones(len(D)))
    seeds = np.random.SeedSequence(SEED + 3).generate_state(NB)
    B = np.array(pool.map(draw3, seeds, chunksize=10))
    rows = []
    for j, (p0, a) in enumerate(SC):
        row = {"p0_pct": p0, "abs_revision": a}
        for off, nm in ((0, "tel"), (6, "web"), (12, "web_minus_tel")):
            row[nm] = pt[off + j]
            row[nm + "_lo"], row[nm + "_hi"] = np.percentile(B[:, off + j], [2.5, 97.5])
        v = B[:, 12 + j]
        row["p_web_minus_tel"] = min(1, 2 * min((v <= 0).mean(), (v >= 0).mean()))
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/tableS12_marginal_by_mode.csv", index=False)
    print(t.round(3).to_string(), flush=True)


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "5"))) as pool:
        if PART in ("1", "all"):
            part1(pool)
        if PART in ("2", "all"):
            part2()
        if PART in ("3", "all"):
            part3(pool)
