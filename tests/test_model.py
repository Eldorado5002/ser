"""AFW (Novelty 1) and MSTC (Novelty 3).

The parameter-neutrality test is what makes the ablation table interpretable:
without it, a base-vs-MSTC delta could be extra capacity rather than
multi-scale receptive fields.
"""
import numpy as np
import pytest
import tensorflow as tf

import config
from model import AdaptiveFeatureWeighting, build_model


@pytest.fixture(scope="module")
def batch():
    r = np.random.default_rng(0)
    return [
        r.standard_normal((4, config.MFCC_LEN)).astype(np.float32),
        r.standard_normal((4, config.ZCR_LEN)).astype(np.float32),
        r.standard_normal((4, config.RMSE_LEN)).astype(np.float32),
    ]


@pytest.mark.parametrize("use_afw", [True, False])
@pytest.mark.parametrize("use_mstc", [True, False])
def test_all_flag_combinations_build_and_predict(use_afw, use_mstc, batch):
    model, weight_model = build_model(use_afw=use_afw, use_mstc=use_mstc)
    out = model.predict(batch, verbose=0)
    assert out.shape == (4, config.NUM_CLASSES)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-5)
    assert (weight_model is not None) == use_afw


def test_afw_weights_are_a_probability_distribution(batch):
    _, weight_model = build_model(use_afw=True, use_mstc=False)
    w = weight_model.predict(batch, verbose=0)
    assert w.shape == (4, 3)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-5)
    assert np.all(w >= 0)


def test_afw_uniform_weights_leave_streams_unchanged():
    """The x3 scaling means the equal-importance solution (1/3,1/3,1/3) is the
    identity - this is what makes base-vs-AFW a clean comparison.

    Zeroing the gate layer forces logits to 0, hence a uniform softmax.
    """
    layer = AdaptiveFeatureWeighting()
    streams = [tf.ones((2, 5)), tf.ones((2, 3)) * 2.0, tf.ones((2, 3)) * 3.0]
    layer(streams)  # build sublayers

    k, b = layer.gate.get_weights()
    layer.gate.set_weights([np.zeros_like(k), np.zeros_like(b)])

    fused, w = layer(streams)
    np.testing.assert_allclose(w.numpy(), 1.0 / 3.0, rtol=1e-5)
    expected = tf.concat(streams, axis=1).numpy()
    np.testing.assert_allclose(fused.numpy(), expected, rtol=1e-5)


def test_afw_adds_negligible_parameters():
    """Spec Table 3 claims AFW costs <0.1% of the parameter budget."""
    plain, _ = build_model(use_afw=False, use_mstc=False)
    afw, _ = build_model(use_afw=True, use_mstc=False)
    extra = afw.count_params() - plain.count_params()
    assert 0 < extra < 0.001 * plain.count_params()


def test_mstc_is_parameter_neutral():
    """Spec Table 3 claims MSTC is ~parameter-neutral because the first
    stage's filter budget is split across the 3/5/7 branches."""
    single, _ = build_model(use_afw=False, use_mstc=False)
    multi, _ = build_model(use_afw=False, use_mstc=True)
    delta = abs(multi.count_params() - single.count_params())
    assert delta / single.count_params() < 0.01


def test_mstc_splits_the_filter_budget_across_three_kernels():
    multi, _ = build_model(use_afw=False, use_mstc=True)
    names = [l.name for l in multi.layers]
    for k in config.MSTC_KERNELS:
        assert f"mstc_k{k}" in names

    total = sum(multi.get_layer(f"mstc_k{k}").filters
                for k in config.MSTC_KERNELS)
    assert total == config.CONV_FILTERS[0]


def test_model_size_is_close_to_the_base_paper():
    """Base paper reports ~7.19 M parameters."""
    model, _ = build_model(use_afw=True, use_mstc=True)
    assert 6.5e6 < model.count_params() < 8.5e6


def test_afw_weights_vary_across_samples(batch):
    """A per-sample gate must not collapse to one global weight vector."""
    _, weight_model = build_model(use_afw=True, use_mstc=False)
    w = weight_model.predict(batch, verbose=0)
    assert np.std(w, axis=0).sum() > 0


def test_model_input_order_is_mfcc_zcr_rmse():
    """utils.StreamScalers.transform emits this order; a mismatch would
    silently feed ZCR into the MFCC input."""
    model, _ = build_model(use_afw=True, use_mstc=False)
    assert [i.name.split(":")[0] for i in model.inputs] == \
           ["mfcc", "zcr", "rmse"]
