"""Marginal slopes (2,000 respondent-cluster bootstrap draws) and full-pipeline bootstrap of the
overlap-weighting contrast (2,000 draws; support, propensity score and weights re-estimated in every draw).
usage: python run2_bootstrap.py <level|log> [NB_MARG] [NB_OVERLAP]"""
import sys
import json
import time
from multiprocessing import Pool
from core import *

OUT = os.environ.get("OUT", "output/revise01")
SCALE = sys.argv[1] if len(sys.argv) > 1 else "level"
NB_M = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
NB_O = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
SEED = 42

D = load(SCALE)
G_ids, INV = np.unique(D.person.values, return_inverse=True)
G = len(G_ids)
XM = design(D, "quad_init")
COLS = list(XM.columns)
XMV, YV = XM.values, D.y.values

# initial expectations 3,4,6 % and revisions +-2, +-5 pp (log scale: same percentages expressed in log points)
if SCALE == "log":
    P0 = {3: 100 * np.log1p(.03), 4: 100 * np.log1p(.04), 6: 100 * np.log1p(.06)}
else:
    P0 = {3: 3., 4: 4., 6: 6.}
SCEN = [(k, v, a) for k, v in P0.items() for a in (-5, -2, 2, 5)]


def wls(Xm, y, w):
    A = Xm.T @ (Xm * w[:, None])
    return np.linalg.solve(A, Xm.T @ (y * w))


def slope(b, p0, x):
    g = lambda n: b[COLS.index(n)]
    s = g("x") + 2 * g("x2") * x + g("xp0") * p0
    if x > 0:
        s += g("x_u") + 2 * g("x2_u") * x + g("xp0_u") * p0
    return s


def marg_draw(seed):
    rng = np.random.default_rng(seed)
    mult = np.bincount(rng.integers(0, G, G), minlength=G)[INV].astype(float)
    b = wls(XMV, YV, mult)
    return [slope(b, p0, a) for _, p0, a in SCEN]


# ---- overlap pipeline on a (re)sampled data set
def overlap_gamma(d, mult):
    m, _ = support_mask(d, mult)
    s, mu = d[m].reset_index(drop=True), mult[m]
    w, _ = propensity(s, mu)
    X = design(s, "base")
    XX = pd.concat([X, X.mul(s.web.values, axis=0).add_suffix("_w")], axis=1)
    b = wls(XX.values, s.y.values, w * mu)
    return b[list(XX.columns).index("x_u_w")], len(s)


def ov_draw(seed):
    rng = np.random.default_rng(seed)
    cnt = np.bincount(rng.integers(0, G, G), minlength=G)
    mult = cnt[INV].astype(float)
    k = mult > 0
    return overlap_gamma(D[k].reset_index(drop=True), mult[k])


if __name__ == "__main__":
    t0 = time.time()
    seeds = np.random.SeedSequence(SEED).generate_state(NB_M + NB_O)
    with Pool(int(os.environ.get("NPROC","12"))) as pool:
        # ---- marginal slopes
        b0 = wls(XMV, YV, np.ones(len(D)))
        pt = [slope(b0, p0, a) for _, p0, a in SCEN]
        boot = np.array(pool.map(marg_draw, seeds[:NB_M], chunksize=20))
        print("marginal done", round(time.time() - t0), flush=True)
        rows = []
        for j, (lab, p0, a) in enumerate(SCEN):
            v = boot[:, j]
            rows.append({"p0_pct": lab, "revision": a, "slope": pt[j], "lo": np.quantile(v, .025), "hi": np.quantile(v, .975)})
        pd.DataFrame(rows).to_csv(f"{OUT}/t4_marginal_slopes_{SCALE}.csv", index=False)
        cr = []
        for lab in P0:
            for a in (2, 5):
                ji, jd = SCEN.index((lab, P0[lab], a)), SCEN.index((lab, P0[lab], -a))
                dv = boot[:, ji] - boot[:, jd]
                cr.append({"p0_pct": lab, "abs_revision": a, "slope_inc": pt[ji], "slope_dec": pt[jd], "diff": pt[ji] - pt[jd],
                           "lo": np.quantile(dv, .025), "hi": np.quantile(dv, .975),
                           "p_boot": min(1, 2 * min((dv <= 0).mean(), (dv >= 0).mean()))})
        cr = pd.DataFrame(cr)
        cr.to_csv(f"{OUT}/t4_contrasts_{SCALE}.csv", index=False)
        print(pd.DataFrame(rows).round(4).to_string())
        print(cr.round(4).to_string(), flush=True)
        # ---- overlap full pipeline
        pt_g, n_s = overlap_gamma(D, np.ones(len(D)))
        res = pool.map(ov_draw, seeds[NB_M:], chunksize=10)
        gb = np.array([r[0] for r in res])
        summ = {"scale": SCALE, "point": float(pt_g), "n_support": int(n_s), "draws": int(len(gb)),
                "ci_lo": float(np.quantile(gb, .025)), "ci_hi": float(np.quantile(gb, .975)),
                "p_boot": float(min(1, 2 * min((gb <= 0).mean(), (gb >= 0).mean()))), "share_neg": float((gb < 0).mean())}
        json.dump(summ, open(f"{OUT}/t5_overlap_bootstrap_{SCALE}.json", "w"), indent=1)
        pd.DataFrame({"gamma_diff": gb}).to_csv(f"{OUT}/t5_overlap_draws_{SCALE}.csv", index=False)
        print(summ, "time", round(time.time() - t0), flush=True)
