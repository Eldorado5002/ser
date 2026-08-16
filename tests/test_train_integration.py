"""End-to-end pipeline wiring.

The critical assertion is that the model saved to best_model.keras is the SAME
model the reported metrics describe. Before the fix, ModelCheckpoint tracked
val_accuracy while EarlyStopping(restore_best_weights=True) tracked val_loss,
so re-running evaluate.py on a finished run could print different numbers than
test_metrics.json.

A deliberately tiny network is used (tiny_model_config): this test exercises
PIPELINE WIRING; architecture is covered by test_model.py.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import config


@pytest.fixture
def synth_metadata(synth_corpus):
    rows = []
    import data_loader as dl
    for corpus, sub, parser in (
        ("RAVDESS", "RAVDESS", dl._parse_ravdess),
        ("TESS", "TESS", dl._parse_tess),
        ("SAVEE", "SAVEE", dl._parse_savee),
        ("CREMA-D", "CREMA-D", dl._parse_cremad),
    ):
        for p in (synth_corpus / sub).rglob("*.wav"):
            emotion = parser(str(p))
            if emotion in config.EMOTION_TO_ID:
                rows.append({"path": str(p.resolve()),
                             "emotion": emotion, "corpus": corpus})
    return pd.DataFrame(rows)


def test_run_experiment_accepts_injected_metadata(synth_metadata, tmp_path,
                                                  monkeypatch,
                                                  tiny_model_config):
    monkeypatch.setattr(config, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    from train import run_experiment

    metrics = run_experiment(use_afw=True, use_eaaa=True, use_mstc=True,
                             use_cadl=True, tag="unit", epochs=2,
                             metadata=synth_metadata)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "macro_f1" in metrics
    assert metrics["config"]["use_afw"] is True


def test_saved_model_reproduces_reported_metrics(synth_metadata, tmp_path,
                                                 monkeypatch,
                                                 tiny_model_config):
    """The regression test for the checkpoint/metrics mismatch."""
    import tensorflow as tf

    runs = tmp_path / "runs"
    monkeypatch.setattr(config, "RUNS_DIR", str(runs))
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    from train import run_experiment
    from data_loader import split_metadata
    from features import build_feature_matrix, df_to_items
    from utils import StreamScalers
    from model import AdaptiveFeatureWeighting

    metrics = run_experiment(use_afw=True, use_eaaa=False, use_mstc=False,
                             use_cadl=False, tag="ckpt", epochs=3,
                             metadata=synth_metadata)

    run_dir = runs / "ckpt"
    saved = json.load(open(run_dir / "test_metrics.json"))
    assert saved["accuracy"] == pytest.approx(metrics["accuracy"])

    _, _, test_df = split_metadata(synth_metadata)
    feats = build_feature_matrix(df_to_items(test_df), desc="test")
    scalers = StreamScalers.load(str(run_dir / "scalers.joblib"))
    x_test = scalers.transform(feats)

    model = tf.keras.models.load_model(
        str(run_dir / "best_model.keras"),
        custom_objects={"AdaptiveFeatureWeighting": AdaptiveFeatureWeighting},
        compile=False)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    reloaded_acc = float((y_pred == feats["y"]).mean())

    assert reloaded_acc == pytest.approx(saved["accuracy"], abs=1e-6)


def test_run_writes_all_expected_artefacts(synth_metadata, tmp_path,
                                           monkeypatch, tiny_model_config):
    runs = tmp_path / "runs"
    monkeypatch.setattr(config, "RUNS_DIR", str(runs))
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    from train import run_experiment

    run_experiment(use_afw=True, use_eaaa=False, use_mstc=False,
                   use_cadl=True, tag="art", epochs=2,
                   metadata=synth_metadata)

    run_dir = runs / "art"
    for name in ("best_model.keras", "scalers.joblib", "training_log.csv",
                 "training_curves.png", "test_metrics.json",
                 "test_classification_report.csv", "test_confusion_matrix.csv",
                 "afw_weights_per_emotion.csv"):
        assert (run_dir / name).exists(), name


def test_metrics_json_contains_the_full_suite(synth_metadata, tmp_path,
                                              monkeypatch, tiny_model_config):
    runs = tmp_path / "runs"
    monkeypatch.setattr(config, "RUNS_DIR", str(runs))
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    from train import run_experiment

    run_experiment(use_afw=False, use_eaaa=False, use_mstc=False,
                   use_cadl=False, tag="suite", epochs=2,
                   metadata=synth_metadata)

    m = json.load(open(runs / "suite" / "test_metrics.json"))
    for key in ("accuracy", "accuracy_95ci", "macro_precision", "macro_recall",
                "macro_f1", "weighted_f1", "specificity_per_class",
                "macro_specificity", "gmean_per_class", "macro_gmean", "mcc",
                "cohen_kappa", "auc_ovr_macro", "confusion_pair_errors",
                "n_test"):
        assert key in m, key
    assert set(m["confusion_pair_errors"]) == {"sad<->neutral", "angry<->fear"}
