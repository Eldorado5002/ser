"""Typeset the project report as a PDF, reading all figures from results/."""
import csv
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, KeepTogether)

ROOT = r"c:\Users\nagas\Desktop\SER"
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
OUT = os.path.join(ROOT, "docs", "SER_Project_Report.pdf")

ORDER = ["base", "afw", "eaaa", "mstc", "cadl", "full"]
LABEL = {"base": "Base (reproduction)", "afw": "+ AFW (N1)",
         "eaaa": "+ EAAA (N2)", "mstc": "+ MSTC (N3)",
         "cadl": "+ CADL (N4)", "full": "Full (all four)"}
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
PAPER, DASUDE = 0.9491, 0.506

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#5c6b7a")
RULE = colors.HexColor("#c6ccd2")
HEADBG = colors.HexColor("#eef1f4")
ACCENT = colors.HexColor("#0b7285")
WARN = colors.HexColor("#c92a2a")
CONTENT_W = A4[0] - 4 * cm


def load(tag, sub="ablation"):
    with open(os.path.join(RES, sub, "runs", tag, "test_metrics.json")) as f:
        return json.load(f)


def load_report(tag):
    path = os.path.join(RES, "ablation", "runs", tag,
                        "test_classification_report.csv")
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row[""] or ""] = row
    return out


M = {t: load(t) for t in ORDER}
R = {t: load_report(t) for t in ORDER}
LD = json.load(open(os.path.join(RES, "leak_dup", "leakage_experiment.json")))
LA = json.load(open(os.path.join(RES, "leak_aug", "augleak_experiment.json")))
base_acc = M["base"]["accuracy"]
n_test = M["base"]["n_test"]

# ----------------------------------------------------------------- styles
ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=19, leading=24, textColor=INK,
                            spaceAfter=6),
    "sub": ParagraphStyle("sub", parent=ss["Normal"], fontSize=10.5,
                          leading=15, alignment=TA_CENTER, textColor=MUTED),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"],
                         fontName="Helvetica-Bold", fontSize=13.5, leading=17,
                         textColor=INK, spaceBefore=16, spaceAfter=7),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"],
                         fontName="Helvetica-Bold", fontSize=11, leading=14,
                         textColor=ACCENT, spaceBefore=11, spaceAfter=5),
    "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=9.6,
                           leading=14.2, textColor=INK, alignment=TA_JUSTIFY,
                           spaceAfter=7),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontSize=8.3, leading=11,
                          textColor=MUTED, alignment=TA_CENTER,
                          spaceBefore=4, spaceAfter=11),
    "note": ParagraphStyle("n", parent=ss["Normal"], fontSize=9.2,
                           leading=13.4, textColor=INK, alignment=TA_JUSTIFY,
                           leftIndent=10, rightIndent=10, spaceBefore=4,
                           spaceAfter=9, borderPadding=7,
                           backColor=colors.HexColor("#f6f8fa"),
                           borderColor=RULE, borderWidth=0.6),
    "code": ParagraphStyle("code", parent=ss["Normal"], fontName="Courier",
                           fontSize=8.2, leading=11.5, textColor=INK,
                           leftIndent=8, spaceAfter=8),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def tbl(data, widths, align=None, head=True, size=8.6):
    t = Table(data, colWidths=widths, hAlign="CENTER", repeatRows=1 if head else 0)
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, INK),
    ]
    if head:
        cmds += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                 ("BACKGROUND", (0, 0), (-1, 0), HEADBG),
                 ("LINEABOVE", (0, 0), (-1, 0), 0.7, INK),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK)]
    if align:
        for col, a in align.items():
            cmds.append(("ALIGN", (col, 0), (col, -1), a))
    t.setStyle(TableStyle(cmds))
    return t


def figure(path, width, caption):
    if not os.path.exists(path):
        return Spacer(1, 1)
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    img = Image(path, width=width, height=width * h / w)
    return KeepTogether([img, P(caption, "cap")])


story = []
A = story.append

# ------------------------------------------------------------- title page
A(Spacer(1, 3.2 * cm))
A(P("Adaptive Feature-Weighted 1D-CNN with Emotion-Aware "
    "Augmentation for Robust Speech Emotion Recognition", "title"))
A(Spacer(1, 0.3 * cm))
A(P("Project Report", "sub"))
A(Spacer(1, 0.15 * cm))
A(P("Domain: Speech Processing &amp; Affective Computing", "sub"))
A(Spacer(1, 0.9 * cm))
A(tbl([["Base paper", "Chourasia, Lamba & Gupta (2026), Scientific Reports"],
       ["Corpora", "RAVDESS + TESS + SAVEE + CREMA-D (12,162 utterances)"],
       ["Classes", "7 (angry, disgust, fear, happy, neutral, sad, surprise)"],
       ["Verified result", "57.95% test accuracy (base configuration)"],
       ["Principal finding", "Two quantified data-leakage mechanisms"]],
      [3.6 * cm, 11.4 * cm], head=False, size=9))
A(Spacer(1, 1.4 * cm))
A(P("<b>Repository:</b> github.com/Eldorado5002/ser", "sub"))
A(PageBreak())

# ---------------------------------------------------------------- abstract
A(P("Abstract", "h1"))
A(P(
    "We implement a lightweight Conv1D speech-emotion-recognition framework "
    "extending a published 1D-CNN baseline with four novelties: adaptive "
    "feature weighting (AFW), emotion-aware adaptive augmentation (EAAA), "
    "multi-scale temporal convolution (MSTC) and a confusion-aware "
    "discriminative loss (CADL). All four are evaluated by a component-wise "
    "ablation on a fused four-corpus dataset of 12,162 utterances across "
    "seven emotions &mdash; the ablation the base paper listed as missing "
    "future work."))
A(P(
    f"On a correctly constructed evaluation protocol the base configuration "
    f"reaches <b>{base_acc*100:.2f}%</b> test accuracy against the "
    f"<b>{PAPER*100:.2f}%</b> reported by the base paper. Investigating that "
    f"gap became the project's principal finding. We identify and quantify "
    f"two data-leakage mechanisms that inflate accuracy on this corpus: "
    f"duplicated dataset mirrors (<b>+{LD['delta']*100:.2f}</b> points) and "
    f"augmenting before splitting "
    f"(<b>+{(LA['augment_before_split']-base_acc)*100:.2f}</b> points). Both "
    f"contamination rates were predicted from first principles before "
    f"measurement and confirmed to within 0.2 points. Our verified figure "
    f"exceeds the closest comparable published multi-corpus result "
    f"({DASUDE*100:.1f}%, Dasude et al., 2024) by "
    f"{(base_acc-DASUDE)*100:.1f} points."))

# ----------------------------------------------------------- contributions
A(P("1.  Contributions", "h1"))
for i, t in enumerate([
    "A complete, test-verified implementation of the base architecture plus "
    "four novelties, with a 117-test suite covering every quantitative claim "
    "this report makes (Section 4).",
    "The component-wise ablation the base paper identified as missing future "
    "work (Section 6).",
    "A reproducibility investigation quantifying two leakage mechanisms "
    "responsible for inflated multi-corpus SER accuracies (Section 7) "
    "&mdash; the principal contribution.",
    "An honest multi-corpus baseline exceeding the closest comparable "
    "published result on the same four corpora."], 1):
    A(P(f"<b>{i}.</b>&nbsp;&nbsp;{t}"))

# ------------------------------------------------------------------ method
A(P("2.  Method", "h1"))
A(P(
    f"The backbone follows the base paper: three frame-level acoustic streams "
    f"(20 MFCC, ZCR, RMSE) fused into a 2,376-dimensional sequential input, "
    f"then five Conv1D stages (512-512-256-256-128) with batch normalisation "
    f"and max pooling, a Dense(512) head and a 7-way softmax. Total "
    f"parameters: <b>{M['base']['config']['params']:,}</b>, within 1.9% of "
    f"the base paper's reported 7.19&nbsp;M."))
d_afw = M["afw"]["config"]["params"] - M["base"]["config"]["params"]
d_mstc = M["mstc"]["config"]["params"] - M["base"]["config"]["params"]
A(tbl([["", "Novelty", "Addresses", "Acts at", "Cost"],
       ["N1", "AFW", "equal stream treatment", "feature fusion",
        f"+{d_afw} params"],
       ["N2", "EAAA", "uniform augmentation", "data (training only)", "zero"],
       ["N3", "MSTC", "single-scale kernels", "first conv stage",
        f"{d_mstc:+d} params"],
       ["N4", "CADL", "confusion-blind loss", "training objective",
        "zero at inference"]],
      [0.9 * cm, 1.7 * cm, 4.5 * cm, 4.2 * cm, 3.7 * cm]))
A(Spacer(1, 0.35 * cm))
A(P(
    "AFW summarises each stream by its mean and standard deviation, passes "
    "them through a compact dense layer and a softmax, then rescales each "
    "stream by three times its weight &mdash; so the equal-importance "
    "solution (1/3, 1/3, 1/3) is the identity, keeping the base-versus-AFW "
    "comparison clean. MSTC replaces the first conv stage with parallel "
    "kernels of size 3, 5 and 7, splitting the 512-filter budget 172/170/170 "
    "so parameter count is preserved. CADL is focal cross-entropy "
    "(&gamma;=2) plus a pairwise penalty (&lambda;=0.5) on the sad-neutral "
    "and angry-fear pairs."))

# ----------------------------------------------------------------- dataset
A(P("3.  Dataset", "h1"))
A(P("Four corpora fused into one seven-class dataset, split 72:8:20 "
    "<b>before</b> any augmentation."))
A(tbl([["Corpus", "Utterances", "Notes"],
       ["RAVDESS", "1,440", "calm merged into neutral"],
       ["TESS", "2,800", "2 female speakers"],
       ["SAVEE", "480", "4 male speakers"],
       ["CREMA-D", "7,442", "91 actors; no surprise class"],
       ["Total", "12,162", "train 8,756 / val 973 / test 2,433"]],
      [3.4 * cm, 2.8 * cm, 8.8 * cm], align={1: "RIGHT"}))
A(Spacer(1, 0.3 * cm))
A(tbl([["Emotion"] + EMOTIONS,
       ["Count", "1,923", "1,923", "1,923", "1,923", "1,895", "1,923",
        "652"]],
      [2.0 * cm] + [1.86 * cm] * 7, size=8.2))
A(Spacer(1, 0.35 * cm))
A(Paragraph(
    "<b>Data-integrity note.</b> The RAVDESS and TESS Kaggle mirrors each "
    "ship the corpus twice (RAVDESS as both <font face='Courier'>Actor_01..24/"
    "</font> and <font face='Courier'>audio_speech_actors_01-24/</font>; TESS "
    "as two directories differing only in capitalisation). A naive recursive "
    "scan yields <b>16,402</b> files instead of 12,162. Section 7 shows why "
    "this matters.", S["note"]))

# ---------------------------------------------------------- implementation
A(P("4.  Implementation and verification", "h1"))
A(P(
    "Because the central claims are quantitative, each is backed by an "
    "automated test. The suite runs in roughly 70 seconds and requires "
    "<b>no audio download</b> &mdash; label parsing depends only on "
    "filenames, and the remaining tests use synthetic waveforms."))
A(tbl([["Test module", "Verifies"],
       ["test_config", "fused input length is exactly 2,376; class order"],
       ["test_data_loader", "all four filename conventions, incl. SAVEE "
                            "sa/su precedence"],
       ["test_no_duplicates", "guard fires on the real 16,402-path listing"],
       ["test_split", "72:8:20 ratios; zero train/test path overlap"],
       ["test_augmentation", "per-class policy; length restored after "
                             "stretch"],
       ["test_features", "stream shapes; cache fingerprint invalidation"],
       ["test_model", "AFW normalisation; MSTC parameter neutrality"],
       ["test_losses", "CADL reduces exactly to cross-entropy when off"],
       ["test_train_integration", "saved checkpoint reproduces reported "
                                  "metrics"]],
      [4.3 * cm, 10.7 * cm], size=8.3))
A(Spacer(1, 0.35 * cm))
A(P("<b>117 tests pass</b> on both TensorFlow 2.15/Keras 2 and TensorFlow "
    "2.20/Keras 3."))
A(P(
    "Two equivalence tests deserve emphasis: <i>CADL is identical to "
    "cross-entropy when disabled</i> and <i>MSTC is parameter-neutral</i>. "
    "Without them a difference between base and +CADL could be a "
    "loss-scaling artefact, and a difference between base and +MSTC could be "
    "extra capacity. They are what make the ablation interpretable."))
A(P("Two defects were found and fixed during verification:"))
A(P(
    "<b>Checkpoint/metrics mismatch.</b> ModelCheckpoint monitored "
    "val_accuracy while EarlyStopping(restore_best_weights=True) monitored "
    "val_loss. Aligning them was insufficient: Keras restores best weights "
    "only when early stopping actually fires, so a run completing all epochs "
    "reported final-epoch metrics while the saved file held a different "
    "epoch. The pipeline now reloads the checkpoint explicitly before "
    "evaluation."))
A(P("<b>Duplicate-mirror contamination</b>, described in Section 7."))

# ------------------------------------------------------------------- setup
A(P("5.  Experimental setup", "h1"))
A(P(
    "Adam (lr 1e-3), batch size 32, up to 50 epochs, early stopping on "
    "validation loss (patience 10) and ReduceLROnPlateau. Per-stream "
    "standardisation fitted on the training partition only. Augmentation "
    "expands the training set to 11,700 samples for every configuration, so "
    "training-set size is never a confound. Six configurations were run on "
    "an NVIDIA P100; seed fixed at 42 throughout."))

A(PageBreak())

# ----------------------------------------------------------------- results
A(P("6.  Results", "h1"))
A(P("6.1  Ablation", "h2"))
A(P(f"All figures are on the held-out test set (n = {n_test:,})."))

rows = [["Configuration", "Accuracy", "95% CI", "Macro F1", "MCC", "Kappa",
         "AUC"]]
for t in ORDER:
    m = M[t]
    lo, hi = m["accuracy_95ci"]
    rows.append([LABEL[t], f"{m['accuracy']*100:.2f}%",
                 f"[{lo*100:.2f}, {hi*100:.2f}]", f"{m['macro_f1']:.4f}",
                 f"{m['mcc']:.4f}", f"{m['cohen_kappa']:.4f}",
                 f"{m['auc_ovr_macro']:.4f}"])
A(tbl(rows, [3.9 * cm, 1.9 * cm, 2.7 * cm, 1.8 * cm, 1.6 * cm, 1.6 * cm,
             1.5 * cm],
      align={1: "RIGHT", 2: "CENTER", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT",
             6: "RIGHT"}, size=8.3))
A(Spacer(1, 0.4 * cm))
A(figure(os.path.join(FIG, "ablation_accuracy.png"), CONTENT_W,
         "Figure 1 &mdash; Test accuracy by configuration with 95% "
         "confidence intervals. The intervals overlap substantially."))

A(Paragraph(
    f"<b>Statistical caveat.</b> With n = {n_test:,} the 95% confidence "
    f"interval half-width is approximately &plusmn;2.0 points. The intervals "
    f"for all six configurations overlap substantially, so <b>none of the "
    f"individual novelty gains is statistically significant at this sample "
    f"size</b>. MSTC and CADL are directionally positive and consistent "
    f"across accuracy, macro F1, MCC and kappa, but this ablation should be "
    f"read as indicative rather than conclusive. Establishing significance "
    f"would require repeated runs across seeds.", S["note"]))

A(P(
    "Two results warrant comment. First, EAAA slightly reduces accuracy, "
    "suggesting the emotion-conditioned policy constrains augmentation "
    "diversity more than it improves class fidelity at this expansion "
    "budget. Second, the full configuration is the weakest of the six: "
    "combining all four novelties compounds rather than mitigates the "
    "overfitting described in Section 6.4."))

rows = [["Emotion", "Precision", "Recall", "F1", "Support"]]
for e in EMOTIONS:
    r = R["base"][e]
    rows.append([e, f"{float(r['precision']):.4f}", f"{float(r['recall']):.4f}",
                 f"{float(r['f1-score']):.4f}",
                 f"{int(float(r['support']))}"])
A(KeepTogether([
    P("6.2  Per-class performance (base configuration)", "h2"),
    tbl(rows, [3.4 * cm, 2.9 * cm, 2.9 * cm, 2.9 * cm, 2.4 * cm],
        align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"})]))
A(Spacer(1, 0.3 * cm))
A(P(
    "Surprise achieves the highest F1 despite being the minority class (652 "
    "samples). It is absent from CREMA-D, so every surprise sample comes "
    "from the acoustically cleaner, less variable corpora. Fear is weakest, "
    "consistent with the base paper's own confusion analysis."))

A(P("6.3  Confusion-pair errors", "h2"))
rows = [["Configuration", "sad-neutral", "angry-fear", "Total"]]
for t in ORDER:
    pe = M[t]["confusion_pair_errors"]
    rows.append([LABEL[t], str(pe["sad<->neutral"]), str(pe["angry<->fear"]),
                 str(pe["sad<->neutral"] + pe["angry<->fear"])])
A(tbl(rows, [5.4 * cm, 3.2 * cm, 3.2 * cm, 2.7 * cm],
      align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}))
A(Spacer(1, 0.35 * cm))
A(figure(os.path.join(FIG, "confusion_pairs.png"), CONTENT_W,
         "Figure 2 &mdash; Errors on the two pairs CADL explicitly targets."))

bp = sum(M["base"]["confusion_pair_errors"].values())
cp = sum(M["cadl"]["confusion_pair_errors"].values())
ap = sum(M["afw"]["confusion_pair_errors"].values())
A(P(
    f"CADL reduces targeted confusions from {bp} to {cp} "
    f"({100*(bp-cp)/bp:.1f}% fewer), confirming the mechanism works as "
    f"designed. Unexpectedly, <b>AFW reduces them further still, to {ap}</b> "
    f"({100*(bp-ap)/bp:.1f}% fewer), despite not being designed for that "
    f"purpose. Per-sample stream reweighting appears to help separate "
    f"acoustically similar pairs &mdash; a result worth investigating."))

A(P("6.4  Learning behaviour", "h2"))
A(P(
    "The base configuration reaches <b>98.9% training accuracy against 61.8% "
    "validation accuracy</b> by epoch 14, with validation loss reaching its "
    "minimum at <b>epoch 4</b> and rising monotonically thereafter. The "
    "limiting factor is therefore generalisation, not capacity: 4.85&nbsp;M "
    "of the 7.32&nbsp;M parameters sit in a single Dense layer over a "
    "9,472-dimensional flatten, trained on 11,700 samples."))
A(figure(os.path.join(RES, "ablation", "runs", "base", "training_curves.png"),
         CONTENT_W * 0.92,
         "Figure 3 &mdash; Base configuration training curves. Validation "
         "accuracy plateaus while training accuracy approaches 99%."))

A(P("6.5  AFW interpretability", "h2"))
afw_csv = os.path.join(RES, "ablation", "runs", "full",
                       "afw_weights_per_emotion.csv")
if os.path.exists(afw_csv):
    with open(afw_csv, newline="") as f:
        rd = [r for r in csv.reader(f) if r]
    rows = [["Emotion", "MFCC", "ZCR", "RMSE"]]
    for r in rd[1:]:
        if len(r) >= 4:
            rows.append([r[0], f"{float(r[1]):.3f}", f"{float(r[2]):.3f}",
                         f"{float(r[3]):.3f}"])
    A(tbl(rows, [4.2 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm],
          align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}))
    A(Spacer(1, 0.3 * cm))
A(P(
    "The gate consistently down-weights MFCC relative to ZCR and RMSE, but "
    "the spread across emotions is small (about 0.02), indicating the module "
    "learns a largely global rebalancing rather than strong per-class "
    "specialisation. This is a weaker interpretability result than the "
    "proposal anticipated and is reported as such."))

A(figure(os.path.join(RES, "ablation", "runs", "base",
                      "test_confusion_matrix_norm.png"), CONTENT_W * 0.72,
         "Figure 4 &mdash; Normalised confusion matrix, base configuration."))

A(PageBreak())

# --------------------------------------------------------- reproducibility
A(P("7.  Reproducibility investigation", "h1"))
A(P(
    f"The base configuration reproduces the base paper's <i>pipeline</i> but "
    f"not its <i>result</i>: {base_acc*100:.2f}% against {PAPER*100:.2f}%. "
    f"Because the implementation is test-verified and the split sizes, class "
    f"distribution and parameter count all match the paper exactly, we "
    f"investigated whether the discrepancy is explained by "
    f"evaluation-protocol defects rather than implementation differences. "
    f"Two controlled experiments were run, each changing exactly one thing "
    f"against the verified base pipeline."))

A(P("7.1  Duplicated dataset mirrors", "h2"))
A(P(
    "Scanning the dataset roots rather than the canonical subdirectories "
    "yields 16,402 files. Because the duplicates are byte-identical "
    "recordings and the split is random over utterances, a clip and its twin "
    "land on opposite sides of the boundary with probability "
    "2 &times; 0.2 &times; 0.8 = 0.32. <b>Predicted contamination: ~41%.</b>"))
A(P(
    f"<b>Measured: 40.8%</b> (1,339 of 3,281 test recordings had a "
    f"byte-identical twin in train or validation). Accuracy rose from "
    f"{base_acc*100:.2f}% to {LD['leaky_base']*100:.2f}% &mdash; "
    f"<b>+{LD['delta']*100:.2f} points</b>. This is a conservative floor: "
    f"the duplicated training set already exceeded the augmentation target, "
    f"so this run trained with <i>less</i> augmentation."))

A(P("7.2  Augmenting before splitting", "h2"))
A(P(
    "The canonical ordering error in published SER pipelines is to augment "
    "every utterance and only then split. Each utterance becomes several "
    "near-identical rows; with three rows per utterance a test row has a "
    "sibling in training with probability 1 &minus; 0.2&sup2; = 0.96. This "
    "experiment used the <b>clean, de-duplicated corpus</b>, isolating the "
    "ordering effect. <b>Predicted contamination: 96%.</b>"))
A(P(
    f"<b>Measured: 96.2%</b> (7,019 of 7,298 test rows came from an "
    f"utterance also present in train or validation). Accuracy rose from "
    f"{base_acc*100:.2f}% to {LA['augment_before_split']*100:.2f}% &mdash; "
    f"<b>+{(LA['augment_before_split']-base_acc)*100:.2f} points</b>."))

A(P("7.3  Leakage accounting", "h2"))
A(tbl([["Pipeline", "Rows", "Contamination", "Accuracy", "Delta"],
       ["Correct: split then augment", "12,162", "0%",
        f"{base_acc*100:.2f}%", "&mdash;"],
       ["Duplicate mirrors", "16,402", "40.8%",
        f"{LD['leaky_base']*100:.2f}%", f"+{LD['delta']*100:.2f}"],
       ["Augment before split", "36,486", "96.2%",
        f"{LA['augment_before_split']*100:.2f}%",
        f"+{(LA['augment_before_split']-base_acc)*100:.2f}"],
       ["Base paper reported", "12,162", "not stated",
        f"{PAPER*100:.2f}%", f"+{(PAPER-base_acc)*100:.2f}"]],
      [5.5 * cm, 2.2 * cm, 3.0 * cm, 2.3 * cm, 2.0 * cm],
      align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}))
A(Spacer(1, 0.4 * cm))
A(figure(os.path.join(FIG, "leakage_accounting.png"), CONTENT_W,
         "Figure 5 &mdash; Accuracy under each evaluation protocol. Only the "
         "leftmost bar is uncontaminated."))
A(P(
    f"Both contamination rates were predicted from first principles before "
    f"measurement (41% vs 40.8%; 96% vs 96.2%), which supports the mechanism "
    f"rather than merely a correlation. Neither defect alone reaches "
    f"{PAPER*100:.2f}%, but they are independent and compound: treating them "
    f"as multiplicative on the error rate gives an estimated ~88%, and "
    f"heavier augmentation than the three rows per utterance used here would "
    f"raise that further."))
A(Paragraph(
    f"<b>Conclusion.</b> Published multi-corpus SER accuracies in the mid-90s "
    f"are reproducible on this data only under evaluation protocols that leak "
    f"training material into the test set. Under a correct protocol the same "
    f"architecture achieves {base_acc*100:.2f}%.", S["note"]))

# -------------------------------------------------------------- discussion
A(P("8.  Discussion", "h1"))
A(P("8.1  Comparison with the literature", "h2"))
A(tbl([["Study", "Corpora", "Accuracy"],
       ["Dasude et al. (2024)", "TESS+RAVDESS+SAVEE+CREMA-D",
        f"{DASUDE*100:.1f}%"],
       ["This work (base, verified)", "same four",
        f"{base_acc*100:.2f}%"],
       ["This work (MSTC, best)", "same four",
        f"{M['mstc']['accuracy']*100:.2f}%"],
       ["Chourasia et al. (2026)", "same four", f"{PAPER*100:.2f}%"]],
      [5.6 * cm, 6.6 * cm, 2.8 * cm], align={2: "RIGHT"}))
A(Spacer(1, 0.35 * cm))
A(P(
    f"Against the closest comparable study &mdash; the same four corpora, "
    f"also reporting a combined-corpus figure &mdash; our verified result is "
    f"{(base_acc-DASUDE)*100:.1f} points higher. CREMA-D constitutes 61% of "
    f"the fused dataset and is the hardest of the four; published audio-only "
    f"results on it typically fall in the 60&ndash;75% band, which is "
    f"difficult to reconcile with a 94.91% average across the combination."))

A(P("8.2  Limitations", "h2"))
for t in [
    "<b>Statistical power.</b> Single-seed runs; no novelty gain is "
    f"significant at n = {n_test:,}. Repeated runs across seeds are required.",
    "<b>Speaker-dependent splitting.</b> The split is over utterances, not "
    "speakers, so the same speaker appears in train and test. This "
    "<i>inflates</i> all reported figures, ours included. It is retained "
    "deliberately for comparability with the base paper; a "
    "speaker-independent protocol would be stricter and would lower every "
    "number here.",
    "<b>Overfitting is unaddressed.</b> The 37-point train/validation gap "
    "indicates substantial headroom from regularisation alone.",
    "<b>The combined-leakage experiment was not completed</b>, so the "
    "compounded estimate (~88%) remains an extrapolation."]:
    A(P("&bull;&nbsp;&nbsp;" + t))

A(P("8.3  Future work", "h2"))
A(P("&bull;&nbsp;&nbsp;Repeated-seed runs with significance testing."))
A(P("&bull;&nbsp;&nbsp;Regularisation study: the Dense head holds 66% of all "
    "parameters."))
A(P("&bull;&nbsp;&nbsp;Speaker-independent evaluation as a stricter secondary "
    "protocol."))
A(P("&bull;&nbsp;&nbsp;Investigating why AFW reduces confusion-pair errors "
    "more than CADL."))

# -------------------------------------------------------------- conclusion
A(P("9.  Conclusion", "h1"))
A(P(
    f"We implemented and ablated four lightweight novelties on a fused "
    f"four-corpus SER dataset, delivering the component-wise analysis the "
    f"base paper listed as future work. Three of four novelties improve on "
    f"the baseline directionally, though none significantly at this sample "
    f"size; CADL reduces its targeted confusion pairs by "
    f"{100*(bp-cp)/bp:.1f}% as designed."))
A(P(
    f"The project's principal contribution is methodological. The base "
    f"paper's {PAPER*100:.2f}% could not be reproduced under a verified "
    f"protocol, and we identify two concrete, independently measured leakage "
    f"mechanisms &mdash; duplicated corpus mirrors and augment-before-split "
    f"ordering &mdash; that inflate accuracy on this data by "
    f"{LD['delta']*100:.1f} and "
    f"{(LA['augment_before_split']-base_acc)*100:.1f} points respectively. "
    f"Both contamination rates were predicted before measurement and "
    f"confirmed to within 0.2 points. We therefore report "
    f"{base_acc*100:.2f}% as an honest baseline for this corpus combination, "
    f"exceeding the closest comparable published result by "
    f"{(base_acc-DASUDE)*100:.1f} points."))

# ---------------------------------------------------------------- appendix
A(P("Appendix A &mdash; Artefacts and reproduction", "h1"))
A(Paragraph(
    "results/ablation/runs/{base,afw,eaaa,mstc,cadl,full}/<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;test_metrics.json&nbsp;&nbsp;&nbsp;&nbsp;"
    "full metric suite<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;test_classification_report.csv<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;test_confusion_matrix{.csv,.png}<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;training_curves.png, training_log.csv<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;afw_weights_per_emotion.csv (AFW runs)<br/>"
    "results/leak_dup/&nbsp;&nbsp;duplicate-mirror experiment<br/>"
    "results/leak_aug/&nbsp;&nbsp;augment-before-split experiment",
    S["code"]))
A(P("Reproduction:"))
A(Paragraph(
    "py -3.10 -m venv .venv<br/>"
    ".venv/Scripts/python.exe -m pip install -r requirements-dev.txt<br/>"
    ".venv/Scripts/python.exe -m pytest tests/ -v",
    S["code"]))
A(P(
    "Kaggle notebooks, in order: 01_features (CPU, feature extraction), "
    "02_train (GPU, six configurations), 03_leakage_test, "
    "04_augment_before_split, 05_both_leaks."))
A(P("Code, tests, notebooks and all artefacts: "
    "<b>github.com/Eldorado5002/ser</b>"))


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    if doc.page > 1:
        canvas.line(2 * cm, A4[1] - 1.55 * cm, A4[0] - 2 * cm,
                    A4[1] - 1.55 * cm)
        canvas.setFont("Helvetica", 7.4)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, A4[1] - 1.42 * cm,
                          "Adaptive Feature-Weighted 1D-CNN for Speech "
                          "Emotion Recognition")
        canvas.drawRightString(A4[0] - 2 * cm, 1.25 * cm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(
    OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
    topMargin=2.1 * cm, bottomMargin=2 * cm,
    title="Adaptive Feature-Weighted 1D-CNN for Speech Emotion Recognition",
    author="SER Project")
doc.build(story, onFirstPage=decorate, onLaterPages=decorate)

print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
