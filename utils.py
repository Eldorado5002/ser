"""Shared utilities: reproducibility, per-stream scaling, plotting."""
from __future__ import annotations

import os
import json
import random

import numpy as np

import config


def set_seed(seed: int = config.RANDOM_SEED):
    """Best-effort determinism across python / numpy / tensorflow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Per-stream standardisation (fit on train, applied to val/test)
# ---------------------------------------------------------------------------
class StreamScalers:
    """StandardScaler per feature stream, since MFCC / ZCR / RMSE live on very
    different numeric ranges and must not share statistics."""

    STREAMS = ("mfcc", "zcr", "rmse")

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scalers = {s: StandardScaler() for s in self.STREAMS}

    def fit(self, feats: dict):
        for s in self.STREAMS:
            self.scalers[s].fit(feats[s])
        return self

    def transform(self, feats: dict) -> list:
        """Return model inputs in the order expected by build_model."""
        return [self.scalers[s].transform(feats[s]).astype(np.float32)
                for s in self.STREAMS]

    def save(self, path: str):
        import joblib
        joblib.dump(self.scalers, path)

    @classmethod
    def load(cls, path: str):
        import joblib
        obj = cls.__new__(cls)
        obj.scalers = joblib.load(path)
        return obj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_history(history: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_path: str, normalise: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype(float)
    if normalise:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
    ax.set_xticks(range(config.NUM_CLASSES), config.EMOTIONS, rotation=45)
    ax.set_yticks(range(config.NUM_CLASSES), config.EMOTIONS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            label = f"{cm[i, j]}" if not normalise else f"{data[i, j]:.2f}"
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if data[i, j] > data.max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
