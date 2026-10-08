# Live demo

Predict the emotion of a speech clip, or of your own voice, using the trained
models from the ablation and efficiency studies.

## Setup

```bash
py -3.10 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pip install sounddevice   # only for record.py
```

The weights (`demo/models/*.weights.npz`) are committed, so no download is
needed.

## 1. Predict from files

```bash
# a single clip
.venv/Scripts/python.exe demo/predict.py demo/clips/OAF_back_angry.wav

# every clip in a folder
.venv/Scripts/python.exe demo/predict.py demo/clips
.venv/Scripts/python.exe demo/predict.py demo/clips_hard
```

Output shows the **full probability distribution**, the prediction with its
confidence and margin, whether it matched the true label, and — for the
`full` model — the AFW module's learned per-clip stream weights.

## 2. Predict from the microphone

```bash
.venv/Scripts/python.exe demo/record.py            # one take
.venv/Scripts/python.exe demo/record.py --loop     # keep going
```

Speak from the moment recording starts: the pipeline uses a 0.6 s offset and a
2.5 s window.

## What the two clip sets show

Correct clips and confidence range, per model:

| Folder | Corpus | `best` (default) | `full` | `mstc` |
|---|---|---|---|---|
| `clips/` | TESS | 7/7 · 99.6–100% | 7/7 · 53–98% | 7/7 · 97–100% |
| `clips_hard/` | CREMA-D | 6/8 · 42–100% | 5/8 · 25–69% | 4/8 · 28–67% |

This contrast is the point, and it is worth demonstrating deliberately.

**TESS** is two female speakers recorded in a studio with deliberately
stereotyped delivery, and because the split is speaker-dependent (report
§8.2) the same speaker appears in training. Every model finds it easy.

**CREMA-D** is 91 crowd-sourced actors with natural delivery, and it is 61% of
the fused dataset, so the combined test accuracy is essentially a CREMA-D
figure: `best` scores 99.46% on TESS but 53.67% on CREMA-D (report §9.5).

Eight clips are far too few to rank the models — one clip moves the score by
12.5 points. The full test set is the real comparison: `best` 66.58%, `mstc`
59.27%, `full` 55.82%.

Read `best`'s confidence with care. It was selected at the epoch with the
highest validation *accuracy*, where it is markedly over-confident (report
§9.5): it scores almost every clip at 99–100%, and it gets the disgust clip
wrong at 85%. Every demo clip also had a 72% chance of landing in the
training split, which the model fits at 96% accuracy. The `full` and `mstc`
models, stopped earlier, give more graded distributions — on CREMA-D their
confidence falls to 25–69%.

Recording your own voice is the strictest test of the three — an entirely
unseen speaker, unseen recording conditions. Expect lower confidence.

## Models

| File | Configuration | Test accuracy |
|---|---|---|
| `best.weights.npz` | GlobalAveragePooling head + dropout 0.35/0.55 + L2 1e-4 + 3× augmentation, early stopping on validation accuracy — the report's headline (§9.5), 2.54 M params | **66.58%** |
| `full.weights.npz` | all four novelties (the only one that shows AFW weights) | 55.82% |
| `mstc.weights.npz` | multi-scale convolution only, best of the six ablation runs | 59.27% |

`best` is the default; pick another with `--model full` or `--model mstc`.

## A note on the weight format

The models were trained on Kaggle, where TensorFlow ships **Keras 3**, while
this project pins **Keras 2.15** (see the report, §4). A Keras 3 `.keras` file
cannot be read by Keras 2, so the weights are stored as plain numpy arrays and
loaded into the architecture that `model.py` rebuilds. This was verified to
reproduce the original predictions to within `3e-07`, and has the useful side
effect of being 27 MB instead of 88 MB.

`best` was exported the same way by `notebooks/09_export_winner.ipynb`
(with `TAG = "gap_reg_aug3_va"`), which also refits its scalers from the
cached 3× training features. Loaded locally it reproduces Kaggle's
predictions to within `2e-08` and its reported test accuracy exactly
(1,620 / 2,433 = 66.58%).
