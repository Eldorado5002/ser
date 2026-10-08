# Adaptive Feature-Weighted 1D-CNN with Emotion-Aware Augmentation for Robust SER

Implementation of the project proposal "Adaptive Feature-Weighted 1D-CNN with
Emotion-Aware Augmentation for Robust Speech Emotion Recognition" (base paper:
Chourasia et al., *Scientific Reports*, 2026). Full specification lives in
`PROJECT_SPEC.md` — read that first.

## Results

Held-out test set, n = 2,433 (RAVDESS + TESS + SAVEE + CREMA-D, seven
emotions). Full write-up: [`docs/REPORT.md`](docs/REPORT.md) /
[`docs/SER_Project_Report.pdf`](docs/SER_Project_Report.pdf).

| Model | Test accuracy | 95% CI | Params |
|---|---|---|---|
| Base-paper reproduction | 57.95% | [55.99, 59.91] | 7.32 M |
| Pooling head + regularisation (`gap_reg_aug3`) | 60.79% | [58.85, 62.73] | 2.54 M |
| **+ early stopping on val accuracy (`gap_reg_aug3_va`)** | **66.58%** | **[64.71, 68.46]** | **2.54 M** |
| Base paper, as reported | 94.91% | — | — |

The base paper's 94.91% does not reproduce under a correct protocol; two
measured leakage mechanisms — duplicated dataset mirrors (+14.71 points) and
augmenting before splitting (+24.27) — account for most of the gap (report
§7). The final model is shipped in [`demo/`](demo/README.md).

## Quick start

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. datasets — download and place under data/ (see PROJECT_SPEC.md, Part B.2)
#    data/RAVDESS   data/TESS   data/SAVEE   data/CREMA-D

# 3. sanity check the dataset scan (should report ~12,162 samples)
python data_loader.py

# 4. train the full proposed model (all four novelties)
python train.py --tag full

# 5. train the base-paper reproduction (no novelties)
python train.py --tag base --no-afw --no-eaaa --no-mstc --no-cadl

# 6. run the complete 6-way ablation study
python ablation.py
```

Artefacts (best model, metrics JSON, classification report, confusion
matrices, training curves, AFW interpretability table) are written to
`runs/<tag>/`. The ablation summary lands in `runs/ablation_results.csv`.

## Repository layout

```
config.py        all hyperparameters, paths, emotion classes, novelty settings
data_loader.py   corpus scanning + label parsing (RAVDESS/TESS/SAVEE/CREMA-D)
augmentation.py  EAAA policy (Novelty 2) + uniform baseline augmentation
features.py      MFCC/ZCR/RMSE stream extraction with on-disk caching
model.py         AFW gate (Novelty 1), MSTC block (Novelty 3), Conv1D backbone
losses.py        CADL composite loss (Novelty 4)
train.py         end-to-end pipeline with per-novelty CLI flags
evaluate.py      full metric suite + confusion analysis + AFW interpretability
ablation.py      6-configuration ablation runner
report.py        aggregate runs/ into tables + acceptance checks
notebooks/       Kaggle notebooks 01-09 (features, training, leakage,
                 sweep, early stopping, final evaluation, demo export)
results/         downloaded Kaggle artefacts every report number is read from
tools/           regenerate the report, PDF and figures from results/
demo/            file and microphone demo with the trained models
tests/           pytest suite (117 tests, no dataset download needed)
docs/            report (Markdown + PDF), sweep notes, build plan
```
