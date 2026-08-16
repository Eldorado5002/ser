"""End-to-end training pipeline.

Runs the full workflow of Section 5.7 of the proposal:
  1. data fusion            (data_loader.build_metadata)
  2. preprocessing          (features.load_waveform)
  3. 72:8:20 split          (data_loader.split_metadata, BEFORE augmentation)
  4. EAAA / uniform aug.    (augmentation.plan_augmentation)   [Novelty 2]
  5. feature extraction     (features.build_feature_matrix)
  6. AFW weighting          (model.AdaptiveFeatureWeighting)   [Novelty 1]
  7. MSTC backbone          (model.build_model)                [Novelty 3]
  8. CADL training          (losses.cadl_loss)                 [Novelty 4]
  9. evaluation             (evaluate.evaluate_predictions)

Each novelty can be switched on/off from the CLI, e.g.:

    python train.py --tag full                       # all four novelties
    python train.py --tag base --no-afw --no-eaaa --no-mstc --no-cadl
    python train.py --tag afw_only --no-eaaa --no-mstc --no-cadl
"""
from __future__ import annotations

import os
import argparse

import numpy as np

import config
from data_loader import build_metadata, split_metadata
from augmentation import plan_augmentation
from features import build_feature_matrix, df_to_items
from utils import set_seed, StreamScalers, plot_history, save_json


def run_experiment(use_afw: bool, use_eaaa: bool, use_mstc: bool,
                   use_cadl: bool, tag: str, epochs: int = config.EPOCHS,
                   metadata=None):
    """Train one configuration and return its test metrics dict.

    ``metadata`` lets callers inject an already-scanned corpus DataFrame -
    tests use a small synthetic corpus, and the Kaggle notebook reuses one
    scan across all six configurations. Leave it None to scan ``data/``.
    """
    import tensorflow as tf
    from model import AdaptiveFeatureWeighting, build_model
    from losses import get_loss
    from evaluate import evaluate_predictions, afw_interpretability

    set_seed()
    run_dir = os.path.join(config.RUNS_DIR, tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== experiment '{tag}'  AFW={use_afw} EAAA={use_eaaa} "
          f"MSTC={use_mstc} CADL={use_cadl} ===")

    # ---- steps 1-3: fuse, split (before augmentation - no leakage) --------
    meta = build_metadata() if metadata is None else metadata
    train_df, val_df, test_df = split_metadata(meta)

    # ---- step 4: augmentation plan (training subset only) -----------------
    train_items = plan_augmentation(train_df, emotion_aware=use_eaaa)

    # ---- step 5: feature extraction (cached) ------------------------------
    aug_key = "eaaa" if use_eaaa else "uniform"
    train_feats = build_feature_matrix(train_items, desc=f"train_{aug_key}")
    val_feats = build_feature_matrix(df_to_items(val_df), desc="val")
    test_feats = build_feature_matrix(df_to_items(test_df), desc="test")

    # ---- per-stream standardisation (fit on train only) -------------------
    scalers = StreamScalers().fit(train_feats)
    scalers.save(os.path.join(run_dir, "scalers.joblib"))
    x_train = scalers.transform(train_feats)
    x_val = scalers.transform(val_feats)
    x_test = scalers.transform(test_feats)

    y_train = np.eye(config.NUM_CLASSES)[train_feats["y"]]
    y_val = np.eye(config.NUM_CLASSES)[val_feats["y"]]

    # ---- steps 6-7: model --------------------------------------------------
    model, weight_model = build_model(use_afw=use_afw, use_mstc=use_mstc)
    model.summary(line_length=100)

    # ---- step 8: compile + train ------------------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss=get_loss(use_cadl),
        metrics=["accuracy"])

    ckpt_path = os.path.join(run_dir, "best_model.keras")
    callbacks = [
        # NOTE: monitor val_loss here to match EarlyStopping's
        # restore_best_weights, so the file on disk IS the model whose metrics
        # get reported. Monitoring different quantities in the two callbacks
        # makes evaluate.py disagree with test_metrics.json.
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", mode="min", save_best_only=True,
            verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE, min_lr=config.MIN_LR,
            verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "training_log.csv")),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=config.BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        verbose=2)

    plot_history(history.history, os.path.join(run_dir, "training_curves.png"))

    # ---- reload the checkpointed best model --------------------------------
    # Do NOT rely on EarlyStopping(restore_best_weights=True): Keras only
    # restores the best weights when early stopping actually FIRES. A run that
    # completes all its epochs keeps final-epoch weights in memory while
    # best_model.keras holds the best epoch - so the reported metrics would
    # describe a different model than the one saved to disk, and re-running
    # evaluate.py on the run would print different numbers.
    # Loading explicitly makes "the evaluated model" and "the saved model"
    # the same thing unconditionally. weight_model shares its layer objects
    # with model, so the AFW interpretability output follows automatically.
    if os.path.exists(ckpt_path):
        best = tf.keras.models.load_model(
            ckpt_path,
            custom_objects={
                "AdaptiveFeatureWeighting": AdaptiveFeatureWeighting},
            compile=False)
        model.set_weights(best.get_weights())
        print(f"[train] evaluating the checkpointed best model ({ckpt_path})")

    # ---- step 9: evaluation ------------------------------------------------
    y_prob = model.predict(x_test, batch_size=config.BATCH_SIZE, verbose=0)
    metrics = evaluate_predictions(test_feats["y"], y_prob, out_dir=run_dir)
    metrics["config"] = {"use_afw": use_afw, "use_eaaa": use_eaaa,
                         "use_mstc": use_mstc, "use_cadl": use_cadl,
                         "params": int(model.count_params())}
    save_json(metrics, os.path.join(run_dir, "test_metrics.json"))

    # ---- Novelty-1 interpretability ---------------------------------------
    if use_afw and weight_model is not None:
        afw_interpretability(weight_model, x_test, test_feats["y"], run_dir)

    print(f"[done] artefacts saved under {run_dir}/")
    return metrics


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="full", help="run name under runs/")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    for name in ("afw", "eaaa", "mstc", "cadl"):
        p.add_argument(f"--{name}", dest=name, action="store_true",
                       default=True)
        p.add_argument(f"--no-{name}", dest=name, action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(use_afw=args.afw, use_eaaa=args.eaaa,
                   use_mstc=args.mstc, use_cadl=args.cadl,
                   tag=args.tag, epochs=args.epochs)
