"""Figure 1 of the paper: CoP-derived reference segmentation of one trial (participant S8, interval 04, left limb).

This is the trial that carries the expert's hand-drawn marks (SSSW/SLT 1005, SLT/SSLW 1119, SSLW/GT 1722).
Run:  python paper/make_fig1.py   ->  paper/figures/fig1_segmentation.pdf
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from refine_sub08_h_valleys import SUBJECTS, parse_csv  # noqa: E402

source, graphs = SUBJECTS["Sub08_H"]
times, values, est = parse_csv(source / "sir_21" / "trial_04_l_segmented.csv")
cut = next(r for r in csv.DictReader((graphs / "phase_cuts.csv").open()) if r["record"] == "sir_21" and r["trial"] == "4" and r["side"] == "L")
steps = [r for r in csv.DictReader((graphs / "stance_segments.csv").open()) if r["record"] == "sir_21" and r["trial"] == "4" and r["side"] == "L"]
valleys = [int(r["cop_valley_sample"]) for r in steps] + [int(steps[-1]["next_valley_sample"])]
peaks = [int(r["cop_peak_sample"]) for r in steps]
b = {p: int(cut[f"{p}_start_sample"]) for p in ("GI", "SSSW", "SLT", "SSLW", "GT")}
n = len(times); t = np.arange(n)

plt.rcParams.update({"font.family": "serif", "font.size": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.45), sharex=True, gridspec_kw={"height_ratios": [1, 2.0], "hspace": 0.08})
spans = [("QS", 0, b["GI"]), ("GI", b["GI"], b["SSSW"]), ("SSSW", b["SSSW"], b["SLT"]), ("SLT", b["SLT"], b["SSLW"]),
         ("SSLW", b["SSLW"], b["GT"]), ("GT", b["GT"], n - 1)]
shade = {"QS": "#f2f2f2", "GI": "#fde9d9", "SSSW": "#e5f2e5", "SLT": "#fbe0ea", "SSLW": "#e2ecf8", "GT": "#f2f2f2"}
for ax in axes:
    for name, a, e in spans:
        ax.axvspan(a, e, color=shade[name], lw=0)
    for x in b.values():
        ax.axvline(x, color="#333333", lw=0.7, ls=(0, (3, 2)))
cop, vgrf = values[:, 8], values[:, 9]
axes[0].plot(t, cop, color="#7a4a24", lw=0.6)
axes[0].plot(valleys, cop[valleys], "v", ms=2.4, color="#1f4e8c", label="CoP valley")
axes[0].plot(peaks, cop[peaks], "^", ms=2.4, color="#b2182b", label="CoP peak")
axes[0].set_ylabel("CoP\n(a.u.)")
axes[0].legend(loc="lower right", ncol=1, frameon=False, fontsize=5.2, handletextpad=0.2, borderaxespad=0.1)
fmg = values[:, :8]
colors = plt.cm.viridis(np.linspace(0, 0.9, 8))
for c in range(8):
    x = fmg[:, c]; lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    axes[1].plot(t, 7 - c + 0.85 * (x - lo) / max(hi - lo, 1e-9), color=colors[c], lw=0.45)
axes[1].set_yticks([7 - c + 0.4 for c in range(8)]); axes[1].set_yticklabels([f"S{c + 1}" for c in range(8)], fontsize=5.5)
axes[1].set_ylabel("FMG (scaled)")
axes[1].set_xlabel("Sample (100 Hz)")
axes[1].set_xlim(0, n - 1)
for name, a, e in spans:
    axes[0].text((a + e) / 2, 1.04, name, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5, fontweight="bold")
names = [r["step"] for r in steps]
for k, name in enumerate(names):
    axes[1].text((valleys[k] + valleys[k + 1]) / 2, 1.0, name, transform=axes[1].get_xaxis_transform(), ha="center", va="bottom", fontsize=4.6, color="#444444")
for ax in axes:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(ROOT / "paper" / "figures" / "fig1_segmentation.pdf", bbox_inches="tight")
fig.savefig(ROOT / "paper" / "figures" / "fig1_segmentation.png", bbox_inches="tight", dpi=300)
print("cuts", b, "valleys", valleys)
