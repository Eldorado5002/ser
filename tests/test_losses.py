"""CADL - Confusion-Aware Discriminative Loss (Novelty 4).

The equivalence test (CADL with both terms disabled == categorical
cross-entropy) is what makes the base-vs-CADL ablation row meaningful: without
it, the delta could be a loss-scaling artefact rather than the novelty.
"""
import numpy as np
import tensorflow as tf

import config
from losses import cadl_loss, get_loss

ANGRY, DISGUST, FEAR, HAPPY, NEUTRAL, SAD, SURPRISE = range(7)


def _onehot(idx):
    return np.eye(config.NUM_CLASSES, dtype=np.float32)[[idx]]


def test_reduces_exactly_to_cross_entropy_when_disabled():
    loss = cadl_loss(use_focal=False, use_pairwise=False)
    r = np.random.default_rng(0)
    y_true = np.eye(config.NUM_CLASSES, dtype=np.float32)[r.integers(0, 7, 16)]
    logits = r.standard_normal((16, config.NUM_CLASSES)).astype(np.float32)
    y_pred = tf.nn.softmax(logits).numpy()

    got = loss(tf.constant(y_true), tf.constant(y_pred)).numpy()
    want = tf.keras.losses.categorical_crossentropy(y_true, y_pred).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_get_loss_returns_plain_crossentropy_when_disabled():
    assert get_loss(use_cadl=False) == "categorical_crossentropy"


def test_get_loss_returns_a_callable_when_enabled():
    fn = get_loss(use_cadl=True)
    assert callable(fn)
    assert fn.__name__ == "cadl_loss"


def test_focal_downweights_easy_samples():
    """gamma=2 must shrink the loss of an already-confident correct sample."""
    plain = cadl_loss(use_focal=False, use_pairwise=False)
    focal = cadl_loss(gamma=2.0, use_focal=True, use_pairwise=False)

    y_true = _onehot(HAPPY)
    y_pred = np.full((1, 7), 0.01 / 6, dtype=np.float32)
    y_pred[0, HAPPY] = 0.99

    assert focal(y_true, y_pred).numpy()[0] < plain(y_true, y_pred).numpy()[0]


def test_focal_barely_affects_hard_samples():
    plain = cadl_loss(use_focal=False, use_pairwise=False)
    focal = cadl_loss(gamma=2.0, use_focal=True, use_pairwise=False)

    y_true = _onehot(HAPPY)
    y_pred = np.full((1, 7), 0.9 / 6, dtype=np.float32)
    y_pred[0, HAPPY] = 0.10

    ratio = focal(y_true, y_pred).numpy()[0] / plain(y_true, y_pred).numpy()[0]
    assert ratio > 0.7


def test_pairwise_penalty_punishes_the_confusion_partner():
    """True class 'sad': leaking probability to 'neutral' (its configured
    partner) must cost more than leaking the same mass to 'happy'."""
    loss = cadl_loss(use_focal=False, use_pairwise=True, pair_lambda=0.5)

    y_true = _onehot(SAD)
    to_partner = np.zeros((1, 7), dtype=np.float32)
    to_partner[0, SAD], to_partner[0, NEUTRAL] = 0.6, 0.4

    to_other = np.zeros((1, 7), dtype=np.float32)
    to_other[0, SAD], to_other[0, HAPPY] = 0.6, 0.4

    assert loss(y_true, to_partner).numpy()[0] > loss(y_true, to_other).numpy()[0]


def test_pairwise_penalty_is_symmetric():
    """The angry<->fear pair must be penalised in both directions."""
    loss = cadl_loss(use_focal=False, use_pairwise=True, pair_lambda=0.5)

    a = np.zeros((1, 7), dtype=np.float32)
    a[0, ANGRY], a[0, FEAR] = 0.6, 0.4
    b = np.zeros((1, 7), dtype=np.float32)
    b[0, FEAR], b[0, ANGRY] = 0.6, 0.4

    np.testing.assert_allclose(loss(_onehot(ANGRY), a).numpy(),
                               loss(_onehot(FEAR), b).numpy(), rtol=1e-5)


def test_unpaired_classes_get_no_penalty():
    """'disgust' and 'surprise' are in no configured pair."""
    plain = cadl_loss(use_focal=False, use_pairwise=False)
    paired = cadl_loss(use_focal=False, use_pairwise=True, pair_lambda=0.5)

    y_pred = np.zeros((1, 7), dtype=np.float32)
    y_pred[0, DISGUST], y_pred[0, SURPRISE] = 0.6, 0.4

    np.testing.assert_allclose(paired(_onehot(DISGUST), y_pred).numpy(),
                               plain(_onehot(DISGUST), y_pred).numpy(),
                               rtol=1e-5)


def test_lambda_scales_the_penalty():
    y_true = _onehot(SAD)
    y_pred = np.zeros((1, 7), dtype=np.float32)
    y_pred[0, SAD], y_pred[0, NEUTRAL] = 0.6, 0.4

    base = cadl_loss(use_focal=False, use_pairwise=False)(y_true, y_pred).numpy()[0]
    half = cadl_loss(use_focal=False, use_pairwise=True,
                     pair_lambda=0.5)(y_true, y_pred).numpy()[0]
    full = cadl_loss(use_focal=False, use_pairwise=True,
                     pair_lambda=1.0)(y_true, y_pred).numpy()[0]

    np.testing.assert_allclose(full - base, 2 * (half - base), rtol=1e-4)


def test_loss_is_finite_at_probability_extremes():
    """Clipping must prevent log(0) from producing inf/nan."""
    loss = cadl_loss()
    y_true = _onehot(SAD)
    for pred in (np.zeros((1, 7), dtype=np.float32),
                 np.eye(7, dtype=np.float32)[[SAD]]):
        assert np.all(np.isfinite(loss(y_true, pred).numpy()))


def test_loss_shape_is_per_sample():
    loss = cadl_loss()
    y_true = np.eye(config.NUM_CLASSES, dtype=np.float32)[[0, 1, 2, 3]]
    y_pred = np.full((4, 7), 1 / 7, dtype=np.float32)
    assert loss(y_true, y_pred).numpy().shape == (4,)
