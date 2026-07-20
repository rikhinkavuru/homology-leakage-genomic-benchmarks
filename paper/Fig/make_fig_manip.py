#!/usr/bin/env python3
"""
Figure 3 -- the dose sweep.

The ten-dose sweep on the clean dataset -- forest drop against realised
         leak fraction, with the as-shipped rank annotated and the first
         inversion marked.

Reads construction_manipulation.csv and dose_response.csv only.

Usage:  ./venv/bin/python paper/Fig/make_fig_manip.py
"""
import ast
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results")

BLUE, VERM, GREY, GREEN = "#0072B2", "#D55E00", "#8a8a8a", "#009E73"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
})

man = pd.read_csv(os.path.join(RES, "construction_manipulation.csv")).set_index("condition")
dose = pd.read_csv(os.path.join(RES, "dose_response.csv"))


def ci(s):
    lo, hi = ast.literal_eval(s)
    return float(lo), float(hi)


fig, axB = plt.subplots(figsize=(3.42, 2.55))

# ---------------------------------------------------------------- Panel B
d = dose.sort_values("phi_used").reset_index(drop=True)
phi = d.phi_used.values
drop = d.rf_drop.values
lo = np.array([ci(s)[0] for s in d.rf_drop_ci])
hi = np.array([ci(s)[1] for s in d.rf_drop_ci])
inv = d.ranking_inverts.values.astype(bool)
# Two of the ten rows are the Table 2 arms re-reported rather than fresh trials.
reused = np.isin(np.round(phi, 4), [round(float(man.loc["A_control_merged_consensus"].leak_jaccard_0p9), 4),
                                    round(float(man.loc["A_manipulated_matched"].leak_jaccard_0p9), 4)])

axB.fill_between(phi, lo, hi, color=BLUE, alpha=0.18, lw=0)
axB.plot(phi, drop, color=BLUE, lw=1.2, zorder=3)
axB.scatter(phi[~inv], drop[~inv], color="white", edgecolor=BLUE, s=34, zorder=4, lw=1.2)
axB.scatter(phi[inv], drop[inv], color=VERM, edgecolor=VERM, s=34, zorder=4, lw=1.2)
axB.scatter(phi[reused], drop[reused], facecolor="none", edgecolor="k", s=95,
            zorder=5, lw=0.9)

first = float(phi[inv].min())
axB.axvline(first, color="k", ls="--", lw=1.0, zorder=2)
axB.text(first + 0.006, 0.116, f"first inversion,\n$\\varphi={first + 1e-9:.3f}$", fontsize=6.8, va="top")
phistar = float(d.loc[d.phi_used == first, "phi_star"].iloc[0])
axB.axvline(phistar, color=GREY, ls=":", lw=1.0, zorder=2)
axB.text(phistar - 0.006, 0.116, f"$\\varphi^{{*}}={phistar:.3f}$", fontsize=6.8,
         va="top", ha="right", color="#555555")
axB.axhline(0, color="k", lw=0.7, zorder=1)
axB.set_xlabel("realised leak fraction $\\varphi$ imposed on\nh. ocr ensembl", fontsize=7.5)
axB.set_ylabel("random-forest accuracy drop", fontsize=7.5)
axB.set_xlim(-0.012, 0.225)
axB.set_ylim(-0.058, 0.135)

h2 = [plt.Line2D([], [], marker="o", mfc="white", mec=BLUE, color="none", ms=6,
                 label="ranking preserved"),
      plt.Line2D([], [], marker="o", color=VERM, ls="", ms=6, label="ranking inverts"),
      plt.Line2D([], [], marker="o", mfc="none", mec="k", color="none", ms=9,
                 label="also an arm of Table 2")]
axB.legend(handles=h2, frameon=False, fontsize=6.6, loc="lower right",
           handletextpad=0.4, borderpad=0.1, labelspacing=0.35)

fig.tight_layout(pad=0.3)
out = os.path.join(HERE, "fig_manipulation")
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=300)
pd.DataFrame({"phi": phi, "rf_drop": drop, "ci_lo": lo, "ci_hi": hi,
              "ranking_inverts": inv, "is_table2_arm": reused}).to_csv(
    out + "_data.csv", index=False)
print("wrote", out + ".pdf", "| first inversion", first, "| phi* ", phistar,
      "| reused rows", int(reused.sum()))
