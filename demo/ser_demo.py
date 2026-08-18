"""Shared loading and rendering helpers for the live demo.

The trained weights come from Kaggle, where TensorFlow ships Keras 3, while
the pinned local environment is Keras 2. A Keras 3 `.keras` file cannot be
read by Keras 2, so the weights are exported as plain numpy arrays and loaded
into the architecture that `model.py` builds. Verified to reproduce the
original predictions to within 3e-07.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

SPECS = {
    "full": dict(use_afw=True, use_mstc=True),
    "mstc": dict(use_afw=False, use_mstc=True),
}

# ANSI colours, one per emotion, in config.EMOTIONS order.
COLOUR = {
    "angry": "\033[91m", "disgust": "\033[95m", "fear": "\033[94m",
    "happy": "\033[93m", "neutral": "\033[97m", "sad": "\033[96m",
    "surprise": "\033[92m",
}
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


def load(tag: str = "full"):
    """Return (model, weight_model, scaler_params) for a trained tag."""
    from model import build_model

    wpath = os.path.join(MODELS, f"{tag}.weights.npz")
    spath = os.path.join(MODELS, f"{tag}.scalers.npz")
    if not os.path.exists(wpath):
        raise SystemExit(
            f"Missing {wpath}\n"
            f"Run  python demo/fetch_model.py  to download the trained "
            f"weights, or see demo/README.md.")

    z = np.load(wpath)
    weights = [z[k] for k in sorted(z.files,
                                    key=lambda s: int(s.split("_")[1]))]
    model, weight_model = build_model(**SPECS[tag])
    model.set_weights(weights)

    s = np.load(spath)
    scalers = {name: (s[f"{name}_mean"], s[f"{name}_scale"])
               for name in ("mfcc", "zcr", "rmse")}
    return model, weight_model, scalers


def featurise(wave: np.ndarray, scalers) -> list:
    """Waveform -> the three scaled model inputs."""
    from features import extract_streams, fix_length

    mfcc, zcr, rmse = extract_streams(fix_length(wave))
    out = []
    for name, arr in (("mfcc", mfcc), ("zcr", zcr), ("rmse", rmse)):
        mean, scale = scalers[name]
        out.append(((arr - mean) / scale).astype(np.float32)[None, :])
    return out


def load_clip(path: str) -> np.ndarray:
    from features import load_waveform
    return load_waveform(path)


def bar(value: float, width: int = 34, fill: str = "#") -> str:
    n = int(round(value * width))
    return fill * n + DIM + "." * (width - n) + RESET


def render(probs: np.ndarray, weights=None, truth: str | None = None,
           title: str = "") -> None:
    """Print the probability distribution and, when present, AFW weights."""
    order = np.argsort(probs)[::-1]
    top = config.EMOTIONS[order[0]]

    if title:
        print(f"\n{BOLD}{title}{RESET}")
    print()

    for i in order:
        e = config.EMOTIONS[i]
        p = float(probs[i])
        mark = ""
        if truth is not None and e == truth:
            mark = f"  {DIM}<- true label{RESET}"
        star = f"{COLOUR[e]}*{RESET}" if e == top else " "
        print(f"  {star} {COLOUR[e]}{e:<9}{RESET} {bar(p)} {p*100:5.1f}%{mark}")

    print()
    conf = float(probs[order[0]])
    margin = conf - float(probs[order[1]])
    verdict = f"{BOLD}{COLOUR[top]}{top.upper()}{RESET}"
    print(f"  prediction: {verdict}   confidence {conf*100:.1f}%   "
          f"margin over 2nd {margin*100:.1f} pts")

    if truth is not None:
        hit = (top == truth)
        tag = (f"\033[92mCORRECT{RESET}" if hit else f"\033[91mWRONG{RESET}"
               f" (true: {truth})")
        print(f"  {tag}")

    if weights is not None:
        w = np.asarray(weights).ravel()
        print()
        print(f"  {DIM}AFW learned stream importance for this clip:{RESET}")
        for name, val in zip(("MFCC (spectral)", "ZCR  (temporal)",
                              "RMSE (energy)  "), w):
            print(f"    {name}  {bar(float(val) * 3, 24, '=')} "
                  f"{float(val)*100:4.1f}%")


def label_from_name(path: str) -> str | None:
    """Best-effort true label from a demo filename (TESS convention)."""
    import data_loader as dl
    for parser in (dl._parse_tess, dl._parse_ravdess, dl._parse_savee,
                   dl._parse_cremad):
        try:
            e = parser(path)
        except Exception:
            e = None
        if e in config.EMOTION_TO_ID:
            return e
    return None
