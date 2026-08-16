"""Model architecture.

Implements:
  * AdaptiveFeatureWeighting (AFW, Novelty 1) - a small attention-style gate
    that learns per-sample softmax importance weights over the three feature
    streams (MFCC / ZCR / RMSE) and rescales each stream before fusion.
  * Multi-Scale Temporal Convolution (MSTC, Novelty 3) - the first Conv1D
    stage is replaced by three parallel branches with kernel sizes 3/5/7 whose
    outputs are concatenated channel-wise. The total filter budget of the
    stage is SPLIT across the branches, keeping parameters ~neutral.
  * The base-paper backbone - five Conv1D stages (512-512-256-256-128) with
    batch normalisation and max pooling, a Dense(512) head and 7-way Softmax.

`build_model` assembles any combination via `use_afw` / `use_mstc` flags,
giving the ablation-study structure for free.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

import config


class AdaptiveFeatureWeighting(layers.Layer):
    """Per-sample softmax gate over the three feature streams (Novelty 1).

    Input : list of 3 tensors  [(B, MFCC_LEN), (B, ZCR_LEN), (B, RMSE_LEN)]
    Output: (fused (B, INPUT_LEN), weights (B, 3))

    Each stream is summarised by its mean and standard deviation; a compact
    dense layer followed by a softmax produces one importance weight per
    stream. Weights are multiplied by 3 before scaling so that the "equal
    importance" solution (1/3, 1/3, 1/3) leaves stream magnitudes unchanged,
    which stabilises early training. Adds only a few thousand parameters.
    """

    def __init__(self, hidden_units: int = config.AFW_HIDDEN_UNITS, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.hidden = layers.Dense(hidden_units, activation="relu",
                                   name="afw_hidden")
        self.gate = layers.Dense(3, activation="softmax", name="afw_gate")

    def build(self, input_shape):
        """Build the sublayers explicitly.

        Keras 3 tracks build state and would otherwise mark this layer built
        while its Dense sublayers are still unbuilt, which can break weight
        restoration when a saved model is reloaded - the path every run
        depends on, since train.py evaluates the reloaded checkpoint.
        Each stream contributes two summary statistics (mean, std).
        """
        n_streams = len(input_shape)
        self.hidden.build((None, 2 * n_streams))
        self.gate.build((None, self.hidden_units))
        super().build(input_shape)

    def call(self, streams):
        summaries = []
        for s in streams:
            mean = tf.reduce_mean(s, axis=1, keepdims=True)
            std = tf.math.reduce_std(s, axis=1, keepdims=True)
            summaries.append(tf.concat([mean, std], axis=1))
        h = self.hidden(tf.concat(summaries, axis=1))     # (B, hidden)
        w = self.gate(h)                                  # (B, 3), sums to 1
        scaled = [s * (3.0 * w[:, i:i + 1]) for i, s in enumerate(streams)]
        fused = tf.concat(scaled, axis=1)                 # (B, INPUT_LEN)
        return fused, w

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg


def _mstc_block(x, total_filters: int):
    """Multi-scale first stage (Novelty 3): parallel kernels 3/5/7, filter
    budget split across branches so parameters stay ~equal to the original
    single-scale stage."""
    k = len(config.MSTC_KERNELS)
    base = total_filters // k
    branch_filters = [base] * k
    branch_filters[0] += total_filters - base * k   # absorb the remainder
    branches = []
    for ks, f in zip(config.MSTC_KERNELS, branch_filters):
        b = layers.Conv1D(f, kernel_size=ks, padding="same",
                          activation="relu", name=f"mstc_k{ks}")(x)
        branches.append(b)
    x = layers.Concatenate(axis=-1, name="mstc_concat")(branches)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(config.DROPOUT_CONV)(x)
    return x


def _conv_block(x, filters: int, name: str):
    """Standard base-paper Conv1D stage: Conv -> BN -> MaxPool -> Dropout."""
    x = layers.Conv1D(filters, kernel_size=config.BASE_KERNEL_SIZE,
                      padding="same", activation="relu", name=name)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(config.DROPOUT_CONV)(x)
    return x


def build_model(use_afw: bool, use_mstc: bool):
    """Build the SER model.

    Returns
    -------
    model        : tf.keras.Model mapping the three streams -> 7-way softmax
    weight_model : tf.keras.Model producing the AFW weights (or None when
                   use_afw=False); used for the interpretability analysis.
    """
    in_mfcc = layers.Input(shape=(config.MFCC_LEN,), name="mfcc")
    in_zcr = layers.Input(shape=(config.ZCR_LEN,), name="zcr")
    in_rmse = layers.Input(shape=(config.RMSE_LEN,), name="rmse")
    streams = [in_mfcc, in_zcr, in_rmse]

    weights_tensor = None
    if use_afw:
        fused, weights_tensor = AdaptiveFeatureWeighting(name="afw")(streams)
    else:
        fused = layers.Concatenate(axis=1, name="static_concat")(streams)

    x = layers.Reshape((config.INPUT_LEN, 1), name="to_sequence")(fused)

    # ---- five-stage Conv1D backbone -------------------------------------
    first, *rest = config.CONV_FILTERS
    if use_mstc:
        x = _mstc_block(x, first)
    else:
        x = _conv_block(x, first, name="conv1")
    for i, f in enumerate(rest, start=2):
        x = _conv_block(x, f, name=f"conv{i}")

    # ---- classification head --------------------------------------------
    x = layers.Flatten()(x)
    x = layers.Dense(config.DENSE_UNITS, activation="relu", name="dense")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_DENSE)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax",
                       name="softmax")(x)

    model = models.Model(streams, out, name="ser_afw_mstc_1dcnn")

    weight_model = None
    if use_afw:
        weight_model = models.Model(streams, weights_tensor,
                                    name="afw_weight_extractor")
    return model, weight_model


if __name__ == "__main__":
    m, _ = build_model(use_afw=True, use_mstc=True)
    m.summary()
