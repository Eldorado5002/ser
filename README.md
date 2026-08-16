# Adaptive Feature-Weighted 1D-CNN with Emotion-Aware Augmentation for Robust SER

Implementation of the project proposal "Adaptive Feature-Weighted 1D-CNN with
Emotion-Aware Augmentation for Robust Speech Emotion Recognition" (base paper:
Chourasia et al., *Scientific Reports*, 2026). Full specification lives in
`PROJECT_SPEC.md` — read that first.

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
```
