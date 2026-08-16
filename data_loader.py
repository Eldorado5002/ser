"""Dataset discovery and label parsing for RAVDESS, TESS, SAVEE and CREMA-D.

Builds a single pandas DataFrame with columns:
    path     - absolute path to the .wav file
    emotion  - one of config.EMOTIONS
    corpus   - which dataset the file came from

Filename conventions handled
----------------------------
RAVDESS : 03-01-06-01-02-01-12.wav ; 3rd field is the emotion code
          01 neutral, 02 calm (mapped to neutral), 03 happy, 04 sad,
          05 angry, 06 fearful, 07 disgust, 08 surprised
TESS    : OAF_back_angry.wav / YAF_dog_ps.wav ; last "_" token is the emotion
          ("ps" / "pleasant_surprise" -> surprise)
SAVEE   : DC_a01.wav or a01.wav inside speaker folders ;
          leading letters encode the emotion: a, d, f, h, n, sa, su
CREMA-D : 1091_DFA_ANG_XX.wav ; 3rd "_" token in
          {ANG, DIS, FEA, HAP, NEU, SAD}  (no surprise in CREMA-D)
"""
from __future__ import annotations

import os
import glob
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Per-corpus parsers: return an emotion string from config.EMOTIONS, or None
# ---------------------------------------------------------------------------
_RAVDESS_CODE = {
    "01": "neutral", "02": "neutral",  # calm merged into neutral (7-class setup)
    "03": "happy", "04": "sad", "05": "angry",
    "06": "fear", "07": "disgust", "08": "surprise",
}

_TESS_MAP = {
    "angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
    "neutral": "neutral", "sad": "sad",
    "ps": "surprise", "pleasant_surprise": "surprise", "surprise": "surprise",
}

_SAVEE_MAP = {
    "a": "angry", "d": "disgust", "f": "fear", "h": "happy",
    "n": "neutral", "sa": "sad", "su": "surprise",
}

_CREMAD_MAP = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}


def _parse_ravdess(fname: str) -> str | None:
    parts = os.path.basename(fname).replace(".wav", "").split("-")
    if len(parts) < 3:
        return None
    return _RAVDESS_CODE.get(parts[2])


def _parse_tess(fname: str) -> str | None:
    stem = os.path.basename(fname).replace(".wav", "").lower()
    token = stem.split("_")[-1]
    return _TESS_MAP.get(token)


def _parse_savee(fname: str) -> str | None:
    stem = os.path.basename(fname).replace(".wav", "")
    # Kaggle version prefixes the speaker: "DC_a01" -> take part after "_"
    if "_" in stem:
        stem = stem.split("_")[-1]
    letters = "".join(ch for ch in stem if ch.isalpha()).lower()
    # two-letter codes first so "sa"/"su" are not read as a single letter
    if letters[:2] in _SAVEE_MAP:
        return _SAVEE_MAP[letters[:2]]
    if letters[:1] in _SAVEE_MAP:
        return _SAVEE_MAP[letters[:1]]
    return None


def _parse_cremad(fname: str) -> str | None:
    parts = os.path.basename(fname).replace(".wav", "").split("_")
    if len(parts) < 3:
        return None
    return _CREMAD_MAP.get(parts[2].upper())


_CORPORA = [
    ("RAVDESS", config.RAVDESS_DIR, _parse_ravdess),
    ("TESS", config.TESS_DIR, _parse_tess),
    ("SAVEE", config.SAVEE_DIR, _parse_savee),
    ("CREMA-D", config.CREMAD_DIR, _parse_cremad),
]


def build_metadata(verbose: bool = True) -> pd.DataFrame:
    """Scan the four dataset folders recursively and return the fused metadata.

    Raises a helpful error if no audio is found so the coding agent knows the
    datasets still need to be downloaded into ``data/``.
    """
    rows = []
    for corpus, root, parser in _CORPORA:
        files = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
        kept = 0
        for f in files:
            emotion = parser(f)
            if emotion in config.EMOTION_TO_ID:
                rows.append({"path": os.path.abspath(f),
                             "emotion": emotion,
                             "corpus": corpus})
                kept += 1
        if verbose:
            print(f"[data] {corpus:8s}: found {len(files):5d} wav files, "
                  f"kept {kept:5d} labelled samples (root: {root})")

    if not rows:
        raise FileNotFoundError(
            "No audio files found. Download RAVDESS, TESS, SAVEE and CREMA-D "
            "and place them under the folders configured in config.py "
            f"(DATA_DIR='{config.DATA_DIR}'). See README.md for links and the "
            "expected directory layout."
        )

    df = pd.DataFrame(rows)
    if verbose:
        print(f"[data] combined dataset: {len(df)} samples")
        print(df.groupby(['emotion']).size().to_string())
    return df


def split_metadata(df: pd.DataFrame):
    """Stratified 72 : 8 : 20 train/val/test split (before augmentation)."""
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df, test_size=config.TEST_FRACTION,
        stratify=df["emotion"], random_state=config.RANDOM_SEED)
    train, val = train_test_split(
        train_val, test_size=config.VAL_FRACTION_OF_TRAINVAL,
        stratify=train_val["emotion"], random_state=config.RANDOM_SEED)

    print(f"[split] train={len(train)}  val={len(val)}  test={len(test)} "
          f"(target ratios 72:8:20)")
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


if __name__ == "__main__":
    meta = build_metadata()
    split_metadata(meta)
