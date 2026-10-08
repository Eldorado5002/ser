"""Figures for the efficiency study: per-corpus accuracy, accuracy/size, and
the early-stopping criterion (validation accuracy vs. loss per epoch)."""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"c:\Users\nagas\Desktop\SER"
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

INK = "#1f2933"
MUTED = "#5c6b7a"
GRID = "#d9dde1"
# base is the grey control; the two improved models take categorical slots
# 1 and 2 (blue, orange), validated as a set against the white page.
BASE_C = "#868e96"
WIN_C = "#2a78d6"
FINAL_C = "#eb6834"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
})

F = json.load(open(os.path.join(RES, "final", "final_results.json")))
V = json.load(open(os.path.join(RES, "final_va", "final_results.json")))
WIN, FINAL = F["winner"], V["winner"]
test = {**F["test"], **V["test"]}
params = {**F["params"], **V["params"]}
pc = {c: {**F["per_corpus"][c], **V["per_corpus"][c]} for c in F["per_corpus"]}
MODELS = [("base", BASE_C), (WIN, WIN_C), (FINAL, FINAL_C)]


def tidy(ax):
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------- per-corpus chart
corpora = [c for c in ["TESS", "RAVDESS", "CREMA-D", "SAVEE"] if c in pc]
labels = corpora + ["COMBINED"]
ns = [pc[c]["n"] for c in corpora] + [test["base"]["n_test"]]
x = np.arange(len(labels))
w = 0.27

fig, ax = plt.subplots(figsize=(7.2, 3.8))
for i, (tag, col) in enumerate(MODELS):
    vals = [pc[c][tag] * 100 for c in corpora] + [test[tag]["accuracy"] * 100]
    bars = ax.bar(x + (i - 1) * w, vals, w, color=col, zorder=3,
                  edgecolor="white", linewidth=1.2,
                  label=f"{tag} ({params[tag] / 1e6:.2f} M params)")
    if tag == FINAL:   # label only the final model; the report table has all
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 1.4, f"{v:.1f}",
                    ha="center", fontsize=7.6, color=INK)

ax.set_xticks(x)
ax.set_xticklabels([f"{lab}\nn={n:,}" for lab, n in zip(labels, ns)],
                   fontsize=8.4)
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, 124)
ax.set_yticks(range(0, 101, 20))
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
tidy(ax)
ax.legend(frameon=False, fontsize=7.8, loc="upper right", ncol=3,
          bbox_to_anchor=(1.0, 1.03))
ax.set_title("Per-corpus test accuracy: the combined figure is dominated "
             "by CREMA-D", fontsize=10, pad=12, loc="left")
fig.tight_layout()
p = os.path.join(FIG, "per_corpus.png")
fig.savefig(p, dpi=200)
plt.close(fig)
print("wrote", p)

# ---------------------------------------------------- accuracy vs. size
fig, ax = plt.subplots(figsize=(5.6, 3.6))
pts = [(tag, params[tag] / 1e6, test[tag]["accuracy"] * 100, col)
       for tag, col in MODELS]
for (_, x0, y0, _), (_, x1, y1, _) in zip(pts, pts[1:]):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="#adb5bd", lw=1.3,
                                shrinkA=7, shrinkB=7))
# label positions chosen to clear the markers and the connecting arrows
place = {"base": ((0, 14), "center"), WIN: ((0, -30), "center"),
         FINAL: ((11, -9), "left")}
for name, mp, acc, col in pts:
    ax.scatter([mp], [acc], s=110, color=col, zorder=4, edgecolor="white",
               linewidth=2)
    off, ha = place[name]
    ax.annotate(f"{name}\n{acc:.2f}%  |  {mp:.2f} M", (mp, acc),
                textcoords="offset points", xytext=off,
                ha=ha, fontsize=8.2, color=INK)

d_size = 100 * (1 - params[FINAL] / params["base"])
d_acc = (test[FINAL]["accuracy"] - test["base"]["accuracy"]) * 100
ax.text(0.98, 0.05, f"base to {FINAL}:\n-{d_size:.0f}% parameters, "
        f"+{d_acc:.2f} points", transform=ax.transAxes, ha="right",
        fontsize=8, color=MUTED)
accs = [a for _, _, a, _ in pts]
ax.set_xlabel("Parameters (millions)")
ax.set_ylabel("Test accuracy (%)")
ax.set_xlim(0.8, 8.6)
ax.set_ylim(np.floor(min(accs)) - 2.5, np.ceil(max(accs)) + 3)
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
tidy(ax)
ax.set_title("Smaller and more accurate", fontsize=10, pad=10, loc="left")
fig.tight_layout()
p = os.path.join(FIG, "accuracy_vs_size.png")
fig.savefig(p, dpi=200)
plt.close(fig)
print("wrote", p)

# ------------------------------------------ early-stopping criterion
# Per-epoch history of the final run, parsed from notebook 08's kernel log
# (the notebook trains with verbose=2, one line per epoch).
log = json.load(open(os.path.join(RES, "early_stop",
                                  "fork-of-notebooka259c04403.log"),
                     encoding="utf-8"))
text = "".join(e["data"] for e in log if e["stream_name"] == "stdout")
section = text.split(f"[run] {FINAL} ")[1]
H = {k: [] for k in ("accuracy", "loss", "val_accuracy", "val_loss")}
for line in section.splitlines():
    if " val_accuracy: " in line:
        for k in H:
            H[k].append(float(re.search(rf" - {k}: ([0-9.]+)", line).group(1)))
ep = np.arange(1, len(H["val_loss"]) + 1)
e_loss = int(np.argmin(H["val_loss"])) + 1
e_acc = int(np.argmax(H["val_accuracy"])) + 1

fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1]})
a1.plot(ep, np.array(H["accuracy"]) * 100, color=BASE_C, lw=1.6,
        label="train")
a1.plot(ep, np.array(H["val_accuracy"]) * 100, color=FINAL_C, lw=2,
        label="validation")
a2.plot(ep, H["loss"], color=BASE_C, lw=1.6, label="train")
a2.plot(ep, H["val_loss"], color=FINAL_C, lw=2, label="validation")

for ax, key, scale in ((a1, "val_accuracy", 100), (a2, "val_loss", 1)):
    for e in (e_loss, e_acc):
        v = H[key][e - 1] * scale
        ax.scatter([e], [v], s=46, color=FINAL_C, edgecolor="white",
                   linewidth=2, zorder=5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    tidy(ax)

va_l, va_a = H["val_accuracy"][e_loss - 1] * 100, H["val_accuracy"][e_acc - 1] * 100
a1.annotate(f"stop on val loss\nepoch {e_loss}: {va_l:.2f}%",
            (e_loss, va_l), textcoords="offset points", xytext=(18, -30),
            fontsize=7.8, color=INK,
            arrowprops=dict(arrowstyle="-", color="#adb5bd", lw=0.8))
a1.annotate(f"stop on val accuracy\nepoch {e_acc}: {va_a:.2f}%",
            (e_acc, va_a), textcoords="offset points", xytext=(-10, -40),
            ha="center", fontsize=7.8, color=INK,
            arrowprops=dict(arrowstyle="-", color="#adb5bd", lw=0.8))
a2.annotate(f"val loss minimum (epoch {e_loss});\nit doubles by epoch {e_acc}",
            (e_loss, H["val_loss"][e_loss - 1]), textcoords="data",
            xytext=(e_loss + 9, 1.2), va="center", fontsize=7.8, color=INK,
            arrowprops=dict(arrowstyle="-", color="#adb5bd", lw=0.8))

a1.set_ylabel("Accuracy (%)")
a2.set_ylabel("Loss")
a2.set_xlabel("Epoch")
a1.legend(frameon=False, fontsize=7.8, loc="lower right")
a1.set_title("Validation loss and validation accuracy disagree about the "
             "best epoch", fontsize=9.6, pad=8, loc="left")
fig.tight_layout()
p = os.path.join(FIG, "early_stopping.png")
fig.savefig(p, dpi=200)
plt.close(fig)
print("wrote", p)
print(f"history: {len(ep)} epochs; val-loss minimum at epoch {e_loss} "
      f"({va_l:.2f}% val acc, loss {H['val_loss'][e_loss-1]:.3f}); "
      f"val-acc maximum at epoch {e_acc} ({va_a:.2f}%, "
      f"loss {H['val_loss'][e_acc-1]:.3f})")
