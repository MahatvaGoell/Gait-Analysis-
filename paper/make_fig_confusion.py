"""Figure: row-normalised confusion matrices of the per-cycle SVM (128 FMG features), unseen participants
(LOSO) and unseen walks of known participants.  Run: python paper/make_fig_confusion.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
names = ["GI", "Short", "Long", "S→L", "L→S", "GT"]
panels = [("(a) Unseen participants", "extra_svm_LOSO_confusion.csv"),
          ("(b) Known participants", "extra_svm_within-subject_confusion.csv")]
plt.rcParams.update({"font.family": "serif", "font.size": 5.6})
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw={"wspace": 0.12})
for ax, (title, name) in zip(axes, panels):
    cm = np.loadtxt(ROOT / "results" / name, delimiter=",")
    pct = cm / cm.sum(1, keepdims=True) * 100
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=5.2, color="white" if pct[i, j] > 55 else "#1a1a1a")
    ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(range(6)); ax.set_yticklabels(names if ax is axes[0] else [])
    ax.set_xlabel("Predicted"); ax.set_title(title, fontsize=6)
axes[0].set_ylabel("Reference class")
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02); cb.set_label("% of class")
fig.savefig(ROOT / "paper" / "figures" / "fig_confusion.pdf", bbox_inches="tight")
fig.savefig(ROOT / "paper" / "figures" / "fig_confusion.png", bbox_inches="tight", dpi=300)
