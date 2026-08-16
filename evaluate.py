"""Evaluation with the full base-paper metric suite.

Computes: accuracy (+95% CI), per-class and macro precision / recall / F1,
per-class specificity, G-mean, MCC, Cohen's Kappa, macro one-vs-rest AUC,
the confusion matrix, and the misclassification counts of the two targeted
confusion pairs (sad<->neutral, angry<->fear).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)

import config
from utils import plot_confusion_matrix, save_json


def _specificity_per_class(cm: np.ndarray) -> np.ndarray:
    """Specificity_c = TN_c / (TN_c + FP_c) from the multi-class CM."""
    total = cm.sum()
    spec = np.zeros(cm.shape[0])
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        spec[c] = tn / max(tn + fp, 1)
    return spec


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray,
                         out_dir: str | None = None, prefix: str = "test"):
    """Compute all metrics from integer labels and predicted probabilities.

    Returns the metrics dict; when out_dir is given also writes
    {prefix}_metrics.json, {prefix}_classification_report.csv and the
    confusion-matrix plots.
    """
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))

    acc = accuracy_score(y_true, y_pred)
    n = len(y_true)
    ci_half = 1.96 * np.sqrt(acc * (1 - acc) / n)          # normal-approx 95% CI

    report = classification_report(
        y_true, y_pred, labels=range(config.NUM_CLASSES),
        target_names=config.EMOTIONS, output_dict=True, zero_division=0)

    spec = _specificity_per_class(cm)
    recall_per_class = np.array(
        [report[e]["recall"] for e in config.EMOTIONS])
    gmean_per_class = np.sqrt(np.clip(recall_per_class * spec, 0, None))

    y_true_1h = np.eye(config.NUM_CLASSES)[y_true]
    try:
        auc = roc_auc_score(y_true_1h, y_prob, multi_class="ovr",
                            average="macro")
    except ValueError:
        auc = float("nan")   # e.g. a class missing from y_true

    pair_confusions = {}
    for a, b in config.CONFUSION_PAIRS:
        ia, ib = config.EMOTION_TO_ID[a], config.EMOTION_TO_ID[b]
        pair_confusions[f"{a}<->{b}"] = int(cm[ia, ib] + cm[ib, ia])

    metrics = {
        "accuracy": acc,
        "accuracy_95ci": [acc - ci_half, acc + ci_half],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "specificity_per_class": {e: float(s) for e, s
                                  in zip(config.EMOTIONS, spec)},
        "macro_specificity": float(spec.mean()),
        "gmean_per_class": {e: float(g) for e, g
                            in zip(config.EMOTIONS, gmean_per_class)},
        "macro_gmean": float(gmean_per_class.mean()),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "auc_ovr_macro": auc,
        "confusion_pair_errors": pair_confusions,
        "n_test": int(n),
    }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        save_json(metrics, os.path.join(out_dir, f"{prefix}_metrics.json"))
        pd.DataFrame(report).T.to_csv(
            os.path.join(out_dir, f"{prefix}_classification_report.csv"))
        np.savetxt(os.path.join(out_dir, f"{prefix}_confusion_matrix.csv"),
                   cm, fmt="%d", delimiter=",")
        plot_confusion_matrix(cm, os.path.join(
            out_dir, f"{prefix}_confusion_matrix.png"), normalise=False)
        plot_confusion_matrix(cm, os.path.join(
            out_dir, f"{prefix}_confusion_matrix_norm.png"), normalise=True)

    print(f"[eval] accuracy={acc:.4f}  macroF1={metrics['macro_f1']:.4f}  "
          f"MCC={metrics['mcc']:.4f}  kappa={metrics['cohen_kappa']:.4f}  "
          f"AUC={metrics['auc_ovr_macro']:.4f}")
    print(f"[eval] confusion-pair errors: {pair_confusions}")
    return metrics


def afw_interpretability(weight_model, inputs, y_true: np.ndarray,
                         out_dir: str):
    """Novelty-1 interpretability: mean AFW stream weight per emotion class."""
    w = weight_model.predict(inputs, verbose=0)       # (N, 3)
    df = pd.DataFrame(w, columns=["mfcc", "zcr", "rmse"])
    df["emotion"] = [config.ID_TO_EMOTION[i] for i in y_true]
    table = df.groupby("emotion").mean().reindex(config.EMOTIONS)
    table.to_csv(os.path.join(out_dir, "afw_weights_per_emotion.csv"))
    print("[eval] AFW mean stream weights per emotion:")
    print(table.round(3).to_string())
    return table


if __name__ == "__main__":
    # Standalone re-evaluation of a finished run:
    #   python evaluate.py runs/<tag>
    import sys
    import tensorflow as tf
    from data_loader import build_metadata, split_metadata
    from features import build_feature_matrix, df_to_items
    from utils import StreamScalers
    from model import AdaptiveFeatureWeighting
    from losses import cadl_loss

    run_dir = sys.argv[1]
    meta = build_metadata()
    _, _, test_df = split_metadata(meta)
    test_feats = build_feature_matrix(df_to_items(test_df), desc="test")

    scalers = StreamScalers.load(os.path.join(run_dir, "scalers.joblib"))
    x_test = scalers.transform(test_feats)

    model = tf.keras.models.load_model(
        os.path.join(run_dir, "best_model.keras"),
        custom_objects={"AdaptiveFeatureWeighting": AdaptiveFeatureWeighting,
                        "cadl_loss": cadl_loss()},
        compile=False)
    y_prob = model.predict(x_test, batch_size=config.BATCH_SIZE, verbose=1)
    evaluate_predictions(test_feats["y"], y_prob, out_dir=run_dir)
