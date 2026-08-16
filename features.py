"""Audio loading, preprocessing and hand-crafted feature extraction.

For every utterance three frame-level feature streams are computed
(base-paper feature set):

    MFCC : config.N_MFCC coefficients per frame, flattened  -> MFCC_LEN values
    ZCR  : zero-crossing rate per frame                     -> ZCR_LEN values
    RMSE : root-mean-square energy per frame                -> RMSE_LEN values

The streams are kept SEPARATE (not concatenated here) because the AFW module
(Novelty 1) needs to weight each stream individually before fusion inside the
model. With the default config the fused length is 2160+108+108 = 2376,
matching the base paper's (2376, 1) input.
"""
from __future__ import annotations

import os
import hashlib

import numpy as np
import librosa
from tqdm import tqdm

import config
from augmentation import augment


# ---------------------------------------------------------------------------
# Loading / preprocessing
# ---------------------------------------------------------------------------
def load_waveform(path: str) -> np.ndarray:
    """Load audio: resample to SAMPLE_RATE, skip OFFSET s, keep DURATION s,
    peak-normalise to [-1, 1] and pad/truncate to a fixed length."""
    y, _ = librosa.load(path, sr=config.SAMPLE_RATE,
                        offset=config.OFFSET, duration=config.DURATION)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return fix_length(y)


def fix_length(y: np.ndarray, n: int | None = None) -> np.ndarray:
    """Zero-pad or truncate a waveform to exactly ``n`` samples."""
    n = n or config.N_SAMPLES
    if len(y) >= n:
        return y[:n].astype(np.float32)
    return np.pad(y, (0, n - len(y))).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature streams
# ---------------------------------------------------------------------------
def _fix_frames(x: np.ndarray, n_frames: int) -> np.ndarray:
    """Pad/truncate the frame axis (last axis) to exactly n_frames."""
    if x.shape[-1] >= n_frames:
        return x[..., :n_frames]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, n_frames - x.shape[-1])]
    return np.pad(x, pad)


def extract_streams(y: np.ndarray):
    """Return (mfcc_flat, zcr, rmse) as fixed-length float32 vectors."""
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH)
    rmse = librosa.feature.rms(
        y=y, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH)
    mfcc = librosa.feature.mfcc(
        y=y, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
        n_fft=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH)

    zcr = _fix_frames(zcr.squeeze(0), config.N_FRAMES)
    rmse = _fix_frames(rmse.squeeze(0), config.N_FRAMES)
    mfcc = _fix_frames(mfcc, config.N_FRAMES).T.ravel()   # (frames*n_mfcc,)

    return (mfcc.astype(np.float32),
            zcr.astype(np.float32),
            rmse.astype(np.float32))


# ---------------------------------------------------------------------------
# Batch extraction with on-disk caching
# ---------------------------------------------------------------------------
def _items_fingerprint(items) -> str:
    h = hashlib.md5()
    for it in items:
        h.update(f"{it['path']}|{it['emotion']}|{it['augment']}|"
                 f"{it.get('emotion_aware')}|{it.get('seed')}".encode())
    h.update(f"{config.SAMPLE_RATE}|{config.DURATION}|{config.OFFSET}|"
             f"{config.FRAME_LENGTH}|{config.HOP_LENGTH}|{config.N_MFCC}"
             .encode())
    return h.hexdigest()[:16]


def build_feature_matrix(items, desc: str, use_cache: bool = True):
    """Extract features for a list of items (see augmentation.plan_augmentation).

    Returns a dict with keys:
        mfcc  (N, MFCC_LEN)   zcr (N, ZCR_LEN)   rmse (N, RMSE_LEN)
        y     (N,) integer class ids
    Results are cached in config.CACHE_DIR keyed by an md5 fingerprint of the
    item list + feature settings, so re-runs are instant.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        config.CACHE_DIR, f"{desc}_{_items_fingerprint(items)}.npz")

    if use_cache and os.path.exists(cache_path):
        print(f"[features] cache hit -> {cache_path}")
        z = np.load(cache_path)
        return {k: z[k] for k in ("mfcc", "zcr", "rmse", "y")}

    mfccs, zcrs, rmses, ys = [], [], [], []
    for it in tqdm(items, desc=f"[features] {desc}", unit="clip"):
        y_wave = load_waveform(it["path"])
        if it["augment"]:
            rng = np.random.default_rng(it["seed"])
            y_wave = fix_length(
                augment(y_wave, it["emotion"], it["emotion_aware"], rng))
        m, z, r = extract_streams(y_wave)
        mfccs.append(m); zcrs.append(z); rmses.append(r)
        ys.append(config.EMOTION_TO_ID[it["emotion"]])

    out = {"mfcc": np.stack(mfccs), "zcr": np.stack(zcrs),
           "rmse": np.stack(rmses), "y": np.asarray(ys, dtype=np.int64)}
    np.savez_compressed(cache_path, **out)
    print(f"[features] cached -> {cache_path}  "
          f"(mfcc {out['mfcc'].shape}, zcr {out['zcr'].shape}, "
          f"rmse {out['rmse'].shape})")
    return out


def df_to_items(df):
    """Convert a metadata DataFrame (val/test) into non-augmented items."""
    return [{"path": r.path, "emotion": r.emotion, "augment": False,
             "emotion_aware": False, "seed": 0} for r in df.itertuples()]
