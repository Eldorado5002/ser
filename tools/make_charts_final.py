"""Figures for the efficiency study: per-corpus accuracy and accuracy/size."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"c:\Users\nagas\Desktop\SER"
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

INK = "#1f2933"
GRID = "#d9dde1"
BASE_C = "#868e96"
NEW_C = "#0b7285"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
})

F = json.load(open(os.path.join(RES, "final", "final_results.json")))
pc = F["per_corpus"]
test = F["test"]
params = F["params"]
WIN = F["winner"]

# ------------------------------------------------------- per-corpus chart
corpora = ["TESS", "RAVDESS", "CREMA-D", "SAVEE"]
corpora = [c for c in corpora if c in pc]
x = np.arange(len(corpora) + 1)
labels = corpora + ["COMBINED"]

base_v = [pc[c]["base"] * 100 for c in corpora] + [test["base"]["accuracy"] * 100]
win_v = [pc[c][WIN] * 100 for c in corpora] + [test[WIN]["accuracy"] * 100]
ns = [pc[c]["n"] for c in corpora] + [sum(pc[c]["n"] for c in corpora)]

fig, ax = plt.subplots(figsize=(7.2, 3.7))
w = 0.38
b1 = ax.bar(x - w / 2, base_v, w, label="base (7.32 M params)",
            color=BASE_C, zorder=3)
b2 = ax.bar(x + w / 2, win_v, w, label=f"{WIN} (2.54 M params)",
            color=NEW_C, zorder=3)

for bars, vals in ((b1, base_v), (b2, win_v)):
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 1.4, f"{v:.1f}",
                ha="center", fontsize=7.6, color=INK)

ax.set_xticks(x)
ax.set_xticklabels([f"{lab}\nn={n:,}" for lab, n in zip(labels, ns)],
                   fontsize=8.4)
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, 122)
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2,
          bbox_to_anchor=(1.0, 1.02))
ax.set_title("Per-corpus test accuracy: the combined figure is dominated "
             "by CREMA-D", fontsize=10, pad=10, loc="left")
fig.tight_layout()
p = os.path.join(FIG, "per_corpus.png")
fig.savefig(p, dpi=200)
plt.close(fig)
print("wrote", p)

# ---------------------------------------------------- accuracy vs. size
fig, ax = plt.subplots(figsize=(5.4, 3.4))
pts = [("base", params["base"] / 1e6, test["base"]["accuracy"] * 100, BASE_C),
       (WIN, params[WIN] / 1e6, test[WIN]["accuracy"] * 100, NEW_C)]
for name, mp, acc, col in pts:
    ax.scatter([mp], [acc], s=150, color=col, zorder=4)
    ax.annotate(f"{name}\n{acc:.2f}%  |  {mp:.2f} M",
                (mp, acc), textcoords="offset points",
                xytext=(0, 16), ha="center", fontsize=8.4, color=INK)

ax.annotate("", xy=(pts[1][1], pts[1][2]), xytext=(pts[0][1], pts[0][2]),
            arrowprops=dict(arrowstyle="->", color="#adb5bd", lw=1.4))
ax.text(4.9, 58.6, "-65% parameters\n+2.84 points", fontsize=8.2,
        color="#5c6b7a", ha="center")

ax.set_xlabel("Parameters (millions)")
ax.set_ylabel("Test accuracy (%)")
ax.set_xlim(1.6, 8.4)
ax.set_ylim(55.5, 64.5)
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Smaller and more accurate", fontsize=10, pad=10, loc="left")
fig.tight_layout()
p = os.path.join(FIG, "accuracy_vs_size.png")
fig.savefig(p, dpi=200)
plt.close(fig)
print("wrote", p)
