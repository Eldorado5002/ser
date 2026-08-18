# Live demo

Predict the emotion of a speech clip, or of your own voice, using the trained
models from the ablation study.

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

| Folder | Corpus | Result | Confidence |
|---|---|---|---|
| `clips/` | TESS | 7/7 correct | 77–98% |
| `clips_hard/` | CREMA-D | 5/8 correct | 24–68% |

This contrast is the point, and it is worth demonstrating deliberately.

**TESS** is two female speakers recorded in a studio with deliberately
stereotyped delivery, and because the split is speaker-dependent (report
§8.2) the same speaker appears in training. The model finds it easy.

**CREMA-D** is 91 crowd-sourced actors with natural delivery, and it is 61% of
the fused dataset — this is where the measured 57.95% comes from.

Note that confidence tracks difficulty: the model is at 95%+ when it is right
on TESS and drops to 24–39% on the CREMA-D clips it gets wrong. The
distribution is informative, not just the argmax.

Recording your own voice is the strictest test of the three — an entirely
unseen speaker, unseen recording conditions. Expect lower confidence.

## Models

| File | Configuration | Test accuracy |
|---|---|---|
| `full.weights.npz` | all four novelties (shows AFW weights) | 55.82% |
| `mstc.weights.npz` | multi-scale convolution only, best of six | 59.27% |

Select with `--model mstc`.

## A note on the weight format

The models were trained on Kaggle, where TensorFlow ships **Keras 3**, while
this project pins **Keras 2.15** (see the report, §4). A Keras 3 `.keras` file
cannot be read by Keras 2, so the weights are stored as plain numpy arrays and
loaded into the architecture that `model.py` rebuilds. This was verified to
reproduce the original predictions to within `3e-07`, and has the useful side
effect of being 27 MB instead of 88 MB.
