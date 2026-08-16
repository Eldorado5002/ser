"""Component-wise ablation study (Section 5.6 / Expected Outcome).

Trains six configurations and writes a comparison table:

    base       : no novelties (uniform augmentation, plain cross-entropy)
    +AFW       : base + Adaptive Feature Weighting
    +EAAA      : base + Emotion-Aware Adaptive Augmentation
    +MSTC      : base + Multi-Scale Temporal Convolution
    +CADL      : base + Confusion-Aware Discriminative Loss
    all        : all four novelties combined (the proposed model)

Run:  python ablation.py            (full 50-epoch runs)
      python ablation.py --epochs 15  (quick sanity sweep)
"""
from __future__ import annotations

import os
import argparse

import pandas as pd

import config
from train import run_experiment

CONFIGS = [
    ("base",  dict(use_afw=False, use_eaaa=False, use_mstc=False, use_cadl=False)),
    ("afw",   dict(use_afw=True,  use_eaaa=False, use_mstc=False, use_cadl=False)),
    ("eaaa",  dict(use_afw=False, use_eaaa=True,  use_mstc=False, use_cadl=False)),
    ("mstc",  dict(use_afw=False, use_eaaa=False, use_mstc=True,  use_cadl=False)),
    ("cadl",  dict(use_afw=False, use_eaaa=False, use_mstc=False, use_cadl=True)),
    ("all",   dict(use_afw=True,  use_eaaa=True,  use_mstc=True,  use_cadl=True)),
]


def main(epochs: int):
    rows = []
    for tag, cfg in CONFIGS:
        metrics = run_experiment(tag=f"ablation_{tag}", epochs=epochs, **cfg)
        rows.append({
            "config": tag,
            **{k: v for k, v in cfg.items()},
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mcc": metrics["mcc"],
            "cohen_kappa": metrics["cohen_kappa"],
            "auc": metrics["auc_ovr_macro"],
            "sad<->neutral_errors":
                metrics["confusion_pair_errors"].get("sad<->neutral"),
            "angry<->fear_errors":
                metrics["confusion_pair_errors"].get("angry<->fear"),
            "params": metrics["config"]["params"],
        })
        # incremental save so partial sweeps are never lost
        os.makedirs(config.RUNS_DIR, exist_ok=True)
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(config.RUNS_DIR, "ablation_results.csv"),
                     index=False)
        print("\n==== ablation table so far ====")
        print(table.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = p.parse_args()
    main(args.epochs)
