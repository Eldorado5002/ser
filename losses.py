"""Confusion-Aware Discriminative Loss (CADL, Novelty 4).

Composite objective =
    focal cross-entropy                          (concentrates on hard samples)
  + lambda * pairwise confusion penalty          (extra margin cost on the
                                                  empirically confusable pairs
                                                  sad<->neutral, angry<->fear)

The penalty term for a pair (a, b) is
    y_a * (-log(1 - p_b))  +  y_b * (-log(1 - p_a))
i.e. whenever the true class is one member of a confusable pair, probability
mass assigned to its confusion partner is explicitly punished. Affects
training only - inference cost is unchanged.
"""
from __future__ import annotations

import tensorflow as tf

import config

_EPS = 1e-7


def _pair_indices():
    return [(config.EMOTION_TO_ID[a], config.EMOTION_TO_ID[b])
            for a, b in config.CONFUSION_PAIRS]


def cadl_loss(gamma: float = config.FOCAL_GAMMA,
              pair_lambda: float = config.PAIR_LAMBDA,
              use_focal: bool = True,
              use_pairwise: bool = True):
    """Build the CADL loss function (Keras-compatible closure).

    With use_focal=False and use_pairwise=False this reduces exactly to plain
    categorical cross-entropy, which keeps the ablation comparison clean.
    """
    pairs = _pair_indices()

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, _EPS, 1.0 - _EPS)

        # -- (focal) cross-entropy ------------------------------------------
        ce = -y_true * tf.math.log(y_pred)                    # (B, C)
        if use_focal:
            ce = tf.pow(1.0 - y_pred, gamma) * ce             # focal modulation
        total = tf.reduce_sum(ce, axis=-1)                    # (B,)

        # -- pairwise confusion penalty -------------------------------------
        if use_pairwise and pair_lambda > 0:
            pen = tf.zeros_like(total)
            for a, b in pairs:
                pen += y_true[:, a] * (-tf.math.log(1.0 - y_pred[:, b]))
                pen += y_true[:, b] * (-tf.math.log(1.0 - y_pred[:, a]))
            total = total + pair_lambda * pen

        return total

    loss.__name__ = "cadl_loss"
    return loss


def get_loss(use_cadl: bool):
    """Return CADL when the novelty is enabled, else plain cross-entropy."""
    if use_cadl:
        return cadl_loss()
    return "categorical_crossentropy"
