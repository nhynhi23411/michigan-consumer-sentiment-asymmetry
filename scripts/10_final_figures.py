"""
Figures for the submission package (30 September 2026). All text in English; no titles
inside figures (captions are in the documents). 300 dpi PNG.

  Figure_1.png   Main manuscript: binned sentiment response by direction, telephone vs web
  Figure_S1.png  Sample construction flow
  Figure_S2.png  Official Michigan series vs series rebuilt from the microdata (replaces old Fig. 3B)
  Figure_S3.png  Distribution of within-person changes, telephone vs web
  Figure_S4.png  Lambda across the baseline and seven checks (bootstrap 95% CI)
  Figure_S5.png  Lambda by period and collection mode (bootstrap 95% CI)
  Figure_S6.png  Bootstrap distribution of lambda, baseline
Also writes output/final/monthly_series_check.csv (reconstruction diagnostics).
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "axes.unicode_minus": True,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8, "legend.frameon": False, "mathtext.fontset": "stix",
})
UP, DOWN, INK, GRID = "#C0392B", "#1F6FA8", "#222222", "#E6E6E6"
FIG = "output/figures/final"
import os
os.makedirs(FIG, exist_ok=True)
FIN = "output/final"
R = json.load(open(f"{FIN}/final_results.json"))

# ------------------------------------------------------------------ data
p = pd.read_csv("data/processed/panel_clean.csv")
OK = p.EDUC_t1.between(1, 6) & p.REGION_t1.between(1, 4) & p.AGE_t1.between(18, 97) & (p.INCOME_t1 >= 0)
d = p[OK & (p.delta_px1_w != 0)].copy()
d["x"], d["y"] = d.delta_px1_w, d.delta_sentiment_w
d["up"] = (d.x > 0).astype(float)
d["web"] = (d.METHOD_t1 == 2) & (d.METHOD_t2 == 2)
d["inc"] = d.INCOME_t1 / 1e4
for k in (2, 3, 4):
    d[f"r{k}"] = (d.REGION_t1 == k).astype(float)
CTRL = ["AGE_t1", "inc", "EDUC_t1", "r2", "r3", "r4"]


def fitted_line(sub):
    """OLS of y on x and controls within one direction group; line with controls at sample means."""
    X = sm.add_constant(sub[["x"] + CTRL])
    b = sm.OLS(sub.y, X).fit().params
    a = b["const"] + sum(b[c] * sub[c].mean() for c in CTRL)
    return a, b["x"]


def binned(sub, edges):
    s = sub.copy()
    s["bin"] = pd.cut(s.x, edges)
    g = s.groupby("bin", observed=True).agg(xm=("x", "mean"), ym=("y", "mean"), n=("y", "size"),
                                            se=("y", lambda v: v.std(ddof=1) / np.sqrt(len(v))))
    return g[g.n >= 30]


# ------------------------------------------------------------------ Figure 1
mf = lambda v: f"{v:.3f}".replace("-", "−")
t1 = pd.read_csv(f"{FIN}/table1_main.csv").set_index("label")
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
panels = [(axes[0], d[~d.web], "Telephone pairs, 1980 to 2024", t1.loc["Telephone pairs, 1980-2024"]),
          (axes[1], d[d.web], "Web pairs, 2024 to 2026", t1.loc["Web pairs, 2024-2026"])]
edges_dn = [-22.5, -15.5, -10.5, -7.5, -5.5, -3.5, -2.5, -1.5, -0.5]
edges_up = [0.5, 1.5, 2.5, 3.5, 5.5, 7.5, 10.5, 15.5, 20.5]
for ax, sub, name, row in panels:
    for grp, col, edges, lab in [(0, DOWN, edges_dn, "Expected inflation fell"),
                                 (1, UP, edges_up, "Expected inflation rose")]:
        s = sub[sub.up == grp]
        b = binned(s, edges)
        ax.errorbar(b.xm, b.ym, yerr=1.96 * b.se, fmt="o", ms=5, color=col, ecolor=col, elinewidth=0.9,
                    capsize=2, alpha=0.9, label=lab, zorder=3)
        a, sl = fitted_line(s)
        xs = np.linspace(min(0, s.x.min()), max(0, s.x.max()), 50)
        ax.plot(xs, a + sl * xs, color=col, lw=1.8, zorder=2)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.axvline(0, color="#999999", lw=0.7)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Change in expected inflation, next 12 months, $\Delta$PX1 (points)")
    lam, lo, hi = row["lambda"], row["lambda_ci_lo"], row["lambda_ci_hi"]
    ax.text(0.03, 0.04, f"{name}\n$\\hat\\beta_{{up}}$ = {mf(row['b_up'])}, $\\hat\\beta_{{down}}$ = {mf(row['b_down'])}\n"
                        f"$\\hat\\lambda$ = {lam:.2f} [{lo:.2f}, {hi:.2f}], N = {int(row['pairs']):,}",
            transform=ax.transAxes, fontsize=8.8, va="bottom", ha="left", usetex=False,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))
axes[0].set_ylabel(r"Mean change in sentiment, $\Delta$ICS")
axes[0].text(-0.02, 1.03, "(a)", transform=axes[0].transAxes, fontsize=11)
axes[1].text(-0.02, 1.03, "(b)", transform=axes[1].transAxes, fontsize=11)
axes[1].legend(loc="upper right", fontsize=8.8)
fig.savefig(f"{FIG}/Figure_1.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Figure S1: sample flow
F = R["flow"]
fig, ax = plt.subplots(figsize=(7.2, 7.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")


def box(x, y, w, h, lines, fc):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec="#333333", lw=1))
    for i, t in enumerate(lines):
        ax.text(x, y + h / 2 - 0.27 - i * 0.3, t, ha="center", va="center",
                fontsize=9.6 if i == 0 else 8.8, fontweight="bold" if i == 0 else "normal")


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))


MX, W = 3.6, 6.2
steps = [
    (["Surveys of Consumers microdata, SDA extract (5 Sep 2026)",
      f"{F['raw_records']:,} interview records, Jan 1978 to Feb 2026"], "#DCE9F5"),
    (["Link reinterviews to the earlier interview",
      "(IDPREV, DATEPR) matched to (ID, survey month)",
      f"{F['linked_pairs']:,} pairs six months apart"], "#DCE9F5"),
    (["Valid ICS and PX1 at both interviews",
      f"{F['pairs_valid_dICS_dPX1']:,} pairs; changes winsorized at 1st/99th pct."], "#FDEBD0"),
    (["Expected inflation changed (ΔPX1 ≠ 0)",
      f"{F['nonzero']:,} pairs"], "#FDEBD0"),
    (["Analytic sample",
      f"{F['analytic_pairs']:,} pairs, {F['respondents']:,} respondents",
      f"Increase {int(t1.loc['Baseline, full sample','n_up']):,}; decrease {int(t1.loc['Baseline, full sample','n_down']):,}"],
     "#D5E8D4"),
]
side = [None, None, None, [f"Unchanged expectation", f"{F['zero_change']:,} pairs"],
        [f"Missing age, income,", f"education or region", f"{F['missing_controls_among_nonzero']:,} pairs"]]
ys = [9.7, 8.05, 6.45, 4.95, 3.3]
hs = [0.95, 1.25, 0.95, 0.95, 1.25]
for i, ((lines, fc), y, h) in enumerate(zip(steps, ys, hs)):
    box(MX, y, W, h, lines, fc)
    if i:
        arrow(MX, ys[i - 1] - hs[i - 1] / 2, MX, y + h / 2)
    if side[i]:
        yy = (ys[i - 1] - hs[i - 1] / 2 + y + h / 2) / 2
        box(8.3, yy, 2.9, 0.3 + 0.3 * (1 + len(side[i])), ["Excluded"] + side[i], "#F6D5D3")
        ax.plot([MX, 8.3 - 1.45], [yy, yy], color=INK, lw=1)
box(2.0, 1.35, 3.4, 1.25, ["Telephone pairs", "1980 to 2024", f"{F['telephone_pairs']:,} pairs"], "#EEF3FA")
box(5.8, 1.35, 3.4, 1.25, ["Web pairs", "2024 to 2026", f"{F['web_pairs']:,} pairs"], "#FBEDEF")
ax.plot([MX, MX], [ys[-1] - hs[-1] / 2, 2.3], color=INK, lw=1)
ax.plot([2.0, 5.8], [2.3, 2.3], color=INK, lw=1)
arrow(2.0, 2.3, 2.0, 1.98); arrow(5.8, 2.3, 5.8, 1.98)
fig.savefig(f"{FIG}/Figure_S1.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Figure S2: series reconstruction
raw = pd.read_csv("data/raw/sca_microdata_raw.csv", skipinitialspace=True, usecols=["YYYYMM", "ICS", "PX1", "WT"])
for c in raw:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw["PX1c"] = raw.PX1.where((raw.PX1 != -97) & (raw.PX1 < 96))


def interp_median(v, w):
    """Weighted median of integer answers treated as intervals [v-0.5, v+0.5)."""
    s = pd.Series(w).groupby(v).sum().sort_index()
    cum = s.cumsum()
    half = s.sum() / 2
    k = np.searchsorted(cum.values, half)
    val = s.index[k]
    below = cum.values[k - 1] if k > 0 else 0.0
    return float(val - 0.5 + (half - below) / s.values[k])


rows = []
for ym, g in raw.groupby("YYYYMM"):
    w = g.WT.where(g.WT > 0)
    gi = g.dropna(subset=["ICS"])
    wi = gi.WT.where(gi.WT > 0).fillna(0)
    ics = np.average(gi.ICS, weights=wi) if wi.sum() > 0 else gi.ICS.mean()
    gp = g.dropna(subset=["PX1c"])
    wp = gp.WT.where(gp.WT > 0).fillna(0)
    if wp.sum() == 0:
        wp = pd.Series(1.0, index=gp.index)
    rows.append({"YYYYMM": ym, "ICS_rebuilt": ics,
                 "PX1_median_rebuilt": interp_median(gp.PX1c.values, wp.values),
                 "PX1_mean_rebuilt": float(np.average(gp.PX1c, weights=wp)),
                 "PX1_old_series": g.PX1.mean()})
ms = pd.DataFrame(rows)
mo = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                  "September", "October", "November", "December"], 1)}


def load(f):
    x = pd.read_csv(f)
    x.columns = [c.strip() for c in x.columns]
    x["YYYYMM"] = x.YYYY * 100 + x.iloc[:, 0].str.strip().map(mo)
    return x


off_px = load("data/external/sca_official_tbmpx1px5.csv")[["YYYYMM", "PX_MD"]]
off_ics = load("data/external/sca_official_tbmics.csv")[["YYYYMM", "ICS_ALL"]]
ms = ms.merge(off_px, on="YYYYMM", how="left").merge(off_ics, on="YYYYMM", how="left")
for c in ["PX_MD", "ICS_ALL"]:
    ms[c] = pd.to_numeric(ms[c], errors="coerce")
ms.to_csv(f"{FIN}/monthly_series_check.csv", index=False)
chk = {
    "months": int(ms.dropna(subset=["ICS_ALL"]).shape[0]),
    "ics_corr": float(ms[["ICS_rebuilt", "ICS_ALL"]].corr().iloc[0, 1]),
    "ics_mean_abs_diff": float((ms.ICS_rebuilt - ms.ICS_ALL).abs().mean()),
    "px_median_corr": float(ms[["PX1_median_rebuilt", "PX_MD"]].corr().iloc[0, 1]),
    "px_median_mean_abs_diff": float((ms.PX1_median_rebuilt - ms.PX_MD).abs().mean()),
    "px_mean_corr": float(ms[["PX1_mean_rebuilt", "PX_MD"]].corr().iloc[0, 1]),
    "px_old_corr": float(ms[["PX1_old_series", "PX_MD"]].corr().iloc[0, 1]),
    "px_old_mean_diff": float((ms.PX1_old_series - ms.PX_MD).mean()),
    "px_mean_mean_diff": float((ms.PX1_mean_rebuilt - ms.PX_MD).mean()),
}
json.dump(chk, open(f"{FIN}/series_check.json", "w"), indent=2)
ms["date"] = pd.to_datetime(ms.YYYYMM.astype(int).astype(str), format="%Y%m")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True, gridspec_kw={"hspace": 0.12})
for ax in (a1, a2):
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31"), color="#F2F2F2", zorder=0)
    ax.axvline(pd.Timestamp("2024-04-01"), color="#777777", ls="--", lw=0.9)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
a1.plot(ms.date, ms.ICS_ALL, color="#555555", lw=2.2, label="Official index")
a1.plot(ms.date, ms.ICS_rebuilt, color=DOWN, lw=0.9, label="Weighted mean of respondent scores")
a1.set_ylabel("ICS")
a1.legend(loc="lower left", fontsize=8.5, ncol=2)
a2.plot(ms.date, ms.PX_MD, color="#555555", lw=2.2, label="Official median")
a2.plot(ms.date, ms.PX1_median_rebuilt, color=UP, lw=0.9, label="Weighted interpolated median of PX1")
a2.set_ylabel("Expected price change,\nnext 12 months (%)")
a2.legend(loc="upper right", fontsize=8.5, ncol=2)
a1.text(0.005, 1.02, "(a)", transform=a1.transAxes, fontsize=11)
a2.text(0.005, 1.02, "(b)", transform=a2.transAxes, fontsize=11)
fig.savefig(f"{FIG}/Figure_S2.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Figure S3: distributions by mode
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"wspace": 0.25})
for sub, col, lab in [(d[~d.web], DOWN, "Telephone pairs"), (d[d.web], UP, "Web pairs")]:
    axs[0].hist(sub.x, bins=np.arange(-22.5, 21.5, 1), density=True, histtype="step", lw=1.4, color=col, label=lab)
    axs[1].hist(sub.y, bins=np.arange(-91, 92, 7), density=True, histtype="step", lw=1.4, color=col, label=lab)
axs[0].set_xlabel(r"$\Delta$PX1 (winsorized, zero changes excluded)")
axs[1].set_xlabel(r"$\Delta$ICS (winsorized)")
axs[0].set_ylabel("Density")
axs[0].legend(fontsize=8.5)
for i, ax in enumerate(axs):
    ax.text(-0.02, 1.03, f"({'ab'[i]})", transform=ax.transAxes, fontsize=11)
fig.savefig(f"{FIG}/Figure_S3.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------------ Figures S4, S5: forest plots of lambda
def forest(df, labels, path, width=8.4):
    fig, ax = plt.subplots(figsize=(width, 0.42 * len(df) + 0.9))
    y = np.arange(len(df))[::-1]
    for yy, (_, r) in zip(y, df.iterrows()):
        sig = r.p_gamma < 0.05
        col = UP if sig else "#8C8C8C"
        ax.plot([r.lambda_ci_lo, min(r.lambda_ci_hi, 12)], [yy, yy], color=col, lw=2.2 if sig else 1.6)
        if r.lambda_ci_hi > 12:
            ax.annotate("", xy=(12.2, yy), xytext=(11.6, yy), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
        ax.scatter(min(r["lambda"], 12), yy, s=46, color=col, edgecolors=INK, lw=0.7, zorder=3)
        ax.text(1.01, yy, f"{r['lambda']:.2f} [{r.lambda_ci_lo:.2f}, {r.lambda_ci_hi:.2f}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8.6)
    ax.axvline(1, color="#555555", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 12.4)
    ax.set_xlabel(r"$\hat\lambda = |\hat\beta_{up}| / |\hat\beta_{down}|$ with 95% bootstrap CI resampling respondents")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.text(1.01, len(df) - 0.4, r"$\hat\lambda$ [95% CI]", transform=ax.get_yaxis_transform(), fontsize=8.6,
            fontweight="bold")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


t2 = pd.read_csv(f"{FIN}/table2_robustness.csv")
forest(t2, ["Baseline", "Excluding changes of telephone mode", "Excluding 2020 to 2022",
            "Unchanged pairs in the lower group", "Expectations for 5 to 10 years (PX5)",
            "Dependent variable: ICC", "Dependent variable: ICE", "Survey weight (WT)"], f"{FIG}/Figure_S4.png")
src = pd.read_csv(f"{FIN}/tableS_sources.csv").set_index("label")
order = ["Telephone, first interview in 1980s", "Telephone, first interview in 1990s", "Telephone, first interview in 2000s",
         "Telephone, first interview in 2010s", "Telephone, first interview in 2020-2023"]
sub = pd.concat([src.loc[order].reset_index(), t1.loc[["Telephone pairs, 1980-2024", "Web pairs, 2024-2026"]].reset_index()])
forest(sub, ["Telephone, 1980s", "Telephone, 1990s", "Telephone, 2000s", "Telephone, 2010s", "Telephone, 2020 to 2023",
             "All telephone pairs, 1980 to 2024", "Web pairs, 2024 to 2026"], f"{FIG}/Figure_S5.png")

# ------------------------------------------------------------------ Figure S6: bootstrap distribution
lam = pd.read_csv(f"{FIN}/bootstrap_lambda_baseline.csv")["lambda"]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.hist(lam, bins=60, color=DOWN, alpha=0.8)
b = t1.loc["Baseline, full sample"]
for v, ls in [(b["lambda"], "-"), (b.lambda_ci_lo, "--"), (b.lambda_ci_hi, "--")]:
    ax.axvline(v, color=INK, ls=ls, lw=1)
ax.axvline(1, color=UP, lw=1.2)
ax.set_xlabel(r"$\hat\lambda$ across 2,000 bootstrap draws resampling respondents")
ax.set_ylabel("Draws")
fig.savefig(f"{FIG}/Figure_S6.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(json.dumps(chk, indent=1))
print("figures written to", FIG)
