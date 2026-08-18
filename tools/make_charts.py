"""Generate the two result charts used in the PDF report."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\nagas\Desktop\SER"
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

ORDER = ["base", "afw", "eaaa", "mstc", "cadl", "full"]
NICE = {"base": "Base", "afw": "+AFW", "eaaa": "+EAAA",
        "mstc": "+MSTC", "cadl": "+CADL", "full": "Full"}

INK = "#1f2933"
GRID = "#d9dde1"
BAR = "#4c6ef5"
BAR_HI = "#0b7285"
BAR_NEG = "#b03060"
LEAK = "#e8590c"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def load(tag):
    with open(os.path.join(RES, "ablation", "runs", tag,
                           "test_metrics.json")) as f:
        return json.load(f)


M = {t: load(t) for t in ORDER}
base_acc = M["base"]["accuracy"]

# ---------------------------------------------------------------- chart 1
fig, ax = plt.subplots(figsize=(7.0, 3.4))

accs = [M[t]["accuracy"] * 100 for t in ORDER]
los = [M[t]["accuracy_95ci"][0] * 100 for t in ORDER]
his = [M[t]["accuracy_95ci"][1] * 100 for t in ORDER]
err = [[a - lo for a, lo in zip(accs, los)], [hi - a for a, hi in zip(accs, his)]]

colors = []
for t in ORDER:
    d = M[t]["accuracy"] - base_acc
    colors.append(BAR if t == "base" else (BAR_HI if d > 0 else BAR_NEG))

bars = ax.bar([NICE[t] for t in ORDER], accs, color=colors, width=0.62,
              zorder=3)
ax.errorbar([NICE[t] for t in ORDER], accs, yerr=err, fmt="none",
            ecolor=INK, elinewidth=1.1, capsize=4, zorder=4)

ax.axhline(base_acc * 100, color=INK, linestyle="--", linewidth=0.9,
           zorder=2, alpha=0.55)
ax.text(5.42, base_acc * 100 + 0.15, "base", fontsize=7.5, color=INK,
        alpha=0.75, ha="right")

for b, a in zip(bars, accs):
    ax.text(b.get_x() + b.get_width() / 2, a + 2.15, f"{a:.2f}",
            ha="center", fontsize=8.2, color=INK)

ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(50, 64)
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Ablation: test accuracy with 95% confidence intervals",
             fontsize=10, pad=10, loc="left")
fig.tight_layout()
p1 = os.path.join(FIG, "ablation_accuracy.png")
fig.savefig(p1, dpi=200)
plt.close(fig)
print("wrote", p1)

# ---------------------------------------------------------------- chart 2
leak_dup = json.load(open(os.path.join(RES, "leak_dup",
                                       "leakage_experiment.json")))
leak_aug = json.load(open(os.path.join(RES, "leak_aug",
                                       "augleak_experiment.json")))

labels = ["Correct\npipeline", "Duplicate\nmirrors",
          "Augment before\nsplit", "Base paper\nreported"]
vals = [base_acc * 100, leak_dup["leaky_base"] * 100,
        leak_aug["augment_before_split"] * 100, 94.91]
contam = ["0%", "40.8%", "96.2%", "not stated"]
cols = [BAR_HI, LEAK, LEAK, "#adb5bd"]

fig, ax = plt.subplots(figsize=(7.0, 3.6))
bars = ax.bar(labels, vals, color=cols, width=0.6, zorder=3)

for b, v, c in zip(bars, vals, contam):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v:.2f}%",
            ha="center", fontsize=9, color=INK)
    ax.text(b.get_x() + b.get_width() / 2, 3.0,
            f"leak\n{c}", ha="center", fontsize=7.6, color="white")

ax.axhline(base_acc * 100, color=BAR_HI, linestyle="--", linewidth=1.0,
           alpha=0.6, zorder=2)
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, 105)
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Leakage accounting: accuracy under each evaluation protocol",
             fontsize=10, pad=10, loc="left")
fig.tight_layout()
p2 = os.path.join(FIG, "leakage_accounting.png")
fig.savefig(p2, dpi=200)
plt.close(fig)
print("wrote", p2)

# ---------------------------------------------------------------- chart 3
fig, ax = plt.subplots(figsize=(7.0, 3.0))
pairs = [(M[t]["confusion_pair_errors"]["sad<->neutral"],
          M[t]["confusion_pair_errors"]["angry<->fear"]) for t in ORDER]
sn = [p[0] for p in pairs]
af = [p[1] for p in pairs]
x = range(len(ORDER))
ax.bar(x, sn, width=0.58, color="#4c6ef5", label="sad <-> neutral", zorder=3)
ax.bar(x, af, width=0.58, bottom=sn, color="#f59f00",
       label="angry <-> fear", zorder=3)
for i, (a, b) in enumerate(zip(sn, af)):
    ax.text(i, a + b + 4, str(a + b), ha="center", fontsize=8.4, color=INK)
ax.set_xticks(list(x))
ax.set_xticklabels([NICE[t] for t in ORDER])
ax.set_ylabel("Misclassifications")
ax.set_ylim(0, 215)
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Errors on the two targeted confusion pairs",
             fontsize=10, pad=10, loc="left")
fig.tight_layout()
p3 = os.path.join(FIG, "confusion_pairs.png")
fig.savefig(p3, dpi=200)
plt.close(fig)
print("wrote", p3)
