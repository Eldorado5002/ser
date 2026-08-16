"""Waveform augmentation: uniform (base-paper style) and EAAA (Novelty 2).

EAAA = Emotion-Aware Adaptive Augmentation. Instead of applying the same
techniques to every class, each training sample receives an augmentation
chosen to mirror the natural acoustic variability of its emotion class
(Table 2 of the proposal):

    happy    -> pitch shifting (+/- 2 semitones)
    sad      -> time stretching (0.8 - 1.2x)
    angry    -> noise injection (sigma 0.001 - 0.005 of peak amplitude)
    fear     -> time shifting (+/- 0.2 s) + light noise
    surprise -> time shifting (+/- 0.2 s)
    disgust  -> mild pitch shift (+/- 1 semitone) + mild stretch (0.9 - 1.1x)
    neutral  -> light noise injection only

All functions take and return a 1-D float32 waveform. Length is NOT
guaranteed to be preserved (time stretching changes it); callers must re-fix
the length afterwards (features.fix_length does this).
"""
from __future__ import annotations

import numpy as np
import librosa

import config


# ---------------------------------------------------------------------------
# Primitive augmentations
# ---------------------------------------------------------------------------
def add_noise(y: np.ndarray, sigma_lo: float = 0.001, sigma_hi: float = 0.005,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Additive Gaussian noise, sigma expressed relative to peak amplitude."""
    rng = rng or np.random.default_rng()
    sigma = rng.uniform(sigma_lo, sigma_hi) * max(np.max(np.abs(y)), 1e-8)
    return (y + rng.normal(0.0, sigma, size=y.shape)).astype(np.float32)


def time_shift(y: np.ndarray, max_shift_s: float = 0.2,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Circularly shift the waveform by up to +/- max_shift_s seconds."""
    rng = rng or np.random.default_rng()
    shift = int(rng.uniform(-max_shift_s, max_shift_s) * config.SAMPLE_RATE)
    return np.roll(y, shift).astype(np.float32)


def pitch_shift(y: np.ndarray, lo: float = -2.0, hi: float = 2.0,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Pitch shift by a random number of semitones in [lo, hi]."""
    rng = rng or np.random.default_rng()
    steps = rng.uniform(lo, hi)
    return librosa.effects.pitch_shift(
        y=y, sr=config.SAMPLE_RATE, n_steps=steps).astype(np.float32)


def time_stretch(y: np.ndarray, lo: float = 0.8, hi: float = 1.2,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Time stretch by a random rate in [lo, hi] (length changes!)."""
    rng = rng or np.random.default_rng()
    rate = rng.uniform(lo, hi)
    return librosa.effects.time_stretch(y=y, rate=rate).astype(np.float32)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def _aug_happy(y, rng):    return pitch_shift(y, -2.0, 2.0, rng)
def _aug_sad(y, rng):      return time_stretch(y, 0.8, 1.2, rng)
def _aug_angry(y, rng):    return add_noise(y, 0.001, 0.005, rng)
def _aug_fear(y, rng):     return add_noise(time_shift(y, 0.2, rng), 0.001, 0.003, rng)
def _aug_surprise(y, rng): return time_shift(y, 0.2, rng)
def _aug_disgust(y, rng):  return time_stretch(pitch_shift(y, -1.0, 1.0, rng), 0.9, 1.1, rng)
def _aug_neutral(y, rng):  return add_noise(y, 0.001, 0.003, rng)

# Emotion -> augmentation function (Table 2 of the proposal)
EAAA_POLICY = {
    "happy": _aug_happy,
    "sad": _aug_sad,
    "angry": _aug_angry,
    "fear": _aug_fear,
    "surprise": _aug_surprise,
    "disgust": _aug_disgust,
    "neutral": _aug_neutral,
}

# Base-paper style pool: one of the four techniques picked uniformly at
# random, regardless of the emotion class.
_UNIFORM_POOL = [
    lambda y, rng: add_noise(y, 0.001, 0.005, rng),
    lambda y, rng: time_shift(y, 0.2, rng),
    lambda y, rng: pitch_shift(y, -2.0, 2.0, rng),
    lambda y, rng: time_stretch(y, 0.8, 1.2, rng),
]


def augment(y: np.ndarray, emotion: str, emotion_aware: bool,
            rng: np.random.Generator) -> np.ndarray:
    """Apply one augmentation to waveform ``y``.

    emotion_aware=True  -> EAAA policy (Novelty 2)
    emotion_aware=False -> uniform base-paper policy (ablation baseline)
    """
    if emotion_aware:
        return EAAA_POLICY[emotion](y, rng)
    fn = _UNIFORM_POOL[rng.integers(0, len(_UNIFORM_POOL))]
    return fn(y, rng)


def plan_augmentation(train_df, emotion_aware: bool, seed: int | None = None):
    """Return the training item list: originals + planned augmented copies.

    Each item is a dict {path, emotion, augment(bool), emotion_aware(bool),
    seed(int)}. The per-item seed makes augmentation reproducible while still
    varying between items.

    The number of augmented copies is chosen so that the final training set
    size matches config.TARGET_TRAIN_SIZE (the base paper's expansion budget),
    or doubles the training set when TARGET_TRAIN_SIZE is None.
    """
    seed = config.RANDOM_SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    items = [{"path": r.path, "emotion": r.emotion, "augment": False,
              "emotion_aware": emotion_aware, "seed": 0}
             for r in train_df.itertuples()]

    n_train = len(items)
    if config.TARGET_TRAIN_SIZE is None:
        n_extra = n_train
    else:
        n_extra = max(config.TARGET_TRAIN_SIZE - n_train, 0)

    # sample source rows for the augmented copies (with replacement if needed)
    replace = n_extra > n_train
    chosen = rng.choice(n_train, size=n_extra, replace=replace)
    for i, idx in enumerate(chosen):
        src = items[idx]
        items.append({"path": src["path"], "emotion": src["emotion"],
                      "augment": True, "emotion_aware": emotion_aware,
                      "seed": int(seed + 1000 + i)})

    mode = "EAAA (emotion-aware)" if emotion_aware else "uniform (base paper)"
    print(f"[aug] policy={mode}  originals={n_train}  augmented={n_extra}  "
          f"total={len(items)}")
    return items
