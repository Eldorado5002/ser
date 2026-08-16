"""EAAA - Emotion-Aware Adaptive Augmentation (Novelty 2).

Proves the Table 2 policy is applied per emotion class, that augmentation is
reproducible, and that length-changing techniques are re-fixed before feature
extraction.
"""
import numpy as np
import pandas as pd
import pytest

import config
from augmentation import EAAA_POLICY, augment, plan_augmentation
from features import fix_length


def test_policy_covers_every_emotion_class():
    """Table 2 must map all seven classes - a KeyError here would crash
    training partway through the run."""
    assert set(EAAA_POLICY) == set(config.EMOTIONS)


def test_time_stretch_emotions_change_length(synth_wave, rng):
    """'sad' is mapped to time stretching, which alters waveform length."""
    out = augment(synth_wave, "sad", emotion_aware=True, rng=rng)
    assert len(out) != len(synth_wave)


def test_noise_emotions_preserve_length(synth_wave):
    """'angry' and 'neutral' are mapped to noise injection only."""
    for emotion in ("angry", "neutral"):
        out = augment(synth_wave, emotion, emotion_aware=True,
                      rng=np.random.default_rng(0))
        assert len(out) == len(synth_wave), emotion


def test_noise_injection_stays_close_to_the_original(synth_wave):
    """Noise must add variation without destroying the signal."""
    out = augment(synth_wave, "angry", emotion_aware=True,
                  rng=np.random.default_rng(0))
    assert not np.array_equal(out, synth_wave)
    corr = np.corrcoef(out, synth_wave)[0, 1]
    assert corr > 0.99


def test_time_shift_preserves_length_and_energy(synth_wave):
    """'surprise' is mapped to circular time shifting."""
    out = augment(synth_wave, "surprise", emotion_aware=True,
                  rng=np.random.default_rng(3))
    assert len(out) == len(synth_wave)
    assert np.isclose(np.sum(out ** 2), np.sum(synth_wave ** 2), rtol=1e-4)


def test_augmentation_is_reproducible(synth_wave):
    a = augment(synth_wave, "happy", True, np.random.default_rng(7))
    b = augment(synth_wave, "happy", True, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_different_seeds_give_different_results(synth_wave):
    a = augment(synth_wave, "happy", True, np.random.default_rng(1))
    b = augment(synth_wave, "happy", True, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_uniform_policy_ignores_emotion(synth_wave):
    """With emotion_aware=False the technique is drawn from a fixed pool,
    so the same seed gives the same result regardless of the label."""
    a = augment(synth_wave, "happy", False, np.random.default_rng(5))
    b = augment(synth_wave, "sad", False, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_length_is_restored_after_stretching(synth_wave, rng):
    """features.build_feature_matrix re-fixes length; verify that contract."""
    stretched = augment(synth_wave, "sad", True, rng)
    assert fix_length(stretched).shape == (config.N_SAMPLES,)


@pytest.fixture
def train_df():
    return pd.DataFrame({
        "path": [f"/fake/{i}.wav" for i in range(100)],
        "emotion": [config.EMOTIONS[i % 7] for i in range(100)],
        "corpus": ["FAKE"] * 100,
    })


def test_plan_reaches_the_target_training_size(train_df, monkeypatch):
    monkeypatch.setattr(config, "TARGET_TRAIN_SIZE", 250)
    items = plan_augmentation(train_df, emotion_aware=True)
    assert len(items) == 250
    assert sum(1 for it in items if not it["augment"]) == 100
    assert sum(1 for it in items if it["augment"]) == 150


def test_plan_doubles_when_target_is_none(train_df, monkeypatch):
    monkeypatch.setattr(config, "TARGET_TRAIN_SIZE", None)
    items = plan_augmentation(train_df, emotion_aware=True)
    assert len(items) == 200


def test_originals_are_never_marked_for_augmentation(train_df, monkeypatch):
    monkeypatch.setattr(config, "TARGET_TRAIN_SIZE", 150)
    items = plan_augmentation(train_df, emotion_aware=True)
    originals = [it for it in items if not it["augment"]]
    assert len(originals) == 100
    assert all(it["seed"] == 0 for it in originals)


def test_plan_is_deterministic(train_df, monkeypatch):
    monkeypatch.setattr(config, "TARGET_TRAIN_SIZE", 150)
    a = plan_augmentation(train_df, emotion_aware=True, seed=42)
    b = plan_augmentation(train_df, emotion_aware=True, seed=42)
    assert [it["path"] for it in a] == [it["path"] for it in b]
    assert [it["seed"] for it in a] == [it["seed"] for it in b]


def test_augmented_copies_inherit_their_source_emotion(train_df, monkeypatch):
    monkeypatch.setattr(config, "TARGET_TRAIN_SIZE", 200)
    items = plan_augmentation(train_df, emotion_aware=True)
    by_path = {f"/fake/{i}.wav": config.EMOTIONS[i % 7] for i in range(100)}
    for it in items:
        assert it["emotion"] == by_path[it["path"]]
