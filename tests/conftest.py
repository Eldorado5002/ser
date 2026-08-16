"""Shared test fixtures.

Design note: every fixture here is SYNTHETIC. The test suite must run with no
dataset downloaded, because dataset acquisition happens on Kaggle, not locally.
"""
import os

import numpy as np
import pytest
import soundfile as sf

import config


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def synth_wave(rng):
    """A deterministic 2.5 s waveform at the project sample rate."""
    t = np.linspace(0, config.DURATION, config.N_SAMPLES, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    noise = 0.05 * rng.standard_normal(config.N_SAMPLES)
    return (tone + noise).astype(np.float32)


def _write_wave(path, seconds=3.2, sr=config.SAMPLE_RATE, seed=0):
    """Write a short synthetic wav. Longer than DURATION+OFFSET so that
    load_waveform's offset/duration slicing has real audio to work with."""
    r = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)
    y = 0.4 * np.sin(2 * np.pi * (180 + 40 * (seed % 7)) * t)
    y = y + 0.05 * r.standard_normal(n)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y.astype(np.float32), sr)


# Files per emotion, per corpus. Must be large enough that the 72:8:20 split
# leaves a validation partition with at least NUM_CLASSES samples, or
# StratifiedShuffleSplit raises. 6 gives 168 files -> val ~13 for 7 classes.
_REPS = 6

_TESS_WORDS = ("back", "dog", "bar", "base", "bath", "bean")


@pytest.fixture
def synth_corpus(tmp_path):
    """Build a miniature four-corpus tree with REAL filename conventions.

    _REPS files per emotion per corpus, so every class survives every split.
    Returns the root path to use as config.DATA_DIR.
    """
    root = tmp_path / "data"
    seed = 0

    # RAVDESS: 3rd hyphen field is the emotion code
    for code in ("01", "02", "03", "04", "05", "06", "07", "08"):
        for rep in range(_REPS):
            name = f"03-01-{code}-01-01-{rep + 1:02d}-01.wav"
            _write_wave(str(root / "RAVDESS" / "Actor_01" / name), seed=seed)
            seed += 1

    # TESS: last underscore token is the emotion
    for tok in ("angry", "disgust", "fear", "happy", "neutral", "sad", "ps"):
        for word in _TESS_WORDS:
            name = f"OAF_{word}_{tok}.wav"
            _write_wave(str(root / "TESS" / f"OAF_{tok}" / name), seed=seed)
            seed += 1

    # SAVEE: leading letters of the utterance code
    for code in ("a", "d", "f", "h", "n", "sa", "su"):
        for idx in range(_REPS):
            name = f"DC_{code}{idx + 1:02d}.wav"
            _write_wave(str(root / "SAVEE" / "ALL" / name), seed=seed)
            seed += 1

    # CREMA-D: 3rd underscore token (no surprise class)
    for tok in ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD"):
        for spk in range(_REPS):
            name = f"{1001 + spk}_DFA_{tok}_XX.wav"
            _write_wave(str(root / "CREMA-D" / "AudioWAV" / name), seed=seed)
            seed += 1

    return root


@pytest.fixture
def corpus_dirs(monkeypatch, synth_corpus):
    """Point config's corpus paths at the synthetic tree."""
    monkeypatch.setattr(config, "DATA_DIR", str(synth_corpus))
    monkeypatch.setattr(config, "RAVDESS_DIR", str(synth_corpus / "RAVDESS"))
    monkeypatch.setattr(config, "TESS_DIR", str(synth_corpus / "TESS"))
    monkeypatch.setattr(config, "SAVEE_DIR", str(synth_corpus / "SAVEE"))
    monkeypatch.setattr(config, "CREMAD_DIR", str(synth_corpus / "CREMA-D"))
    import data_loader
    monkeypatch.setattr(data_loader, "_CORPORA", [
        ("RAVDESS", str(synth_corpus / "RAVDESS"), data_loader._parse_ravdess),
        ("TESS", str(synth_corpus / "TESS"), data_loader._parse_tess),
        ("SAVEE", str(synth_corpus / "SAVEE"), data_loader._parse_savee),
        ("CREMA-D", str(synth_corpus / "CREMA-D"), data_loader._parse_cremad),
    ])
    return synth_corpus


@pytest.fixture
def tiny_model_config(monkeypatch):
    """Shrink the network so integration tests run in seconds.

    The architecture itself is covered by test_model.py; integration tests
    exercise PIPELINE WIRING, so a small network is the correct trade.
    """
    monkeypatch.setattr(config, "CONV_FILTERS", [16, 16, 8, 8, 8])
    monkeypatch.setattr(config, "DENSE_UNITS", 32)
    monkeypatch.setattr(config, "TARGET_TRAIN_SIZE", None)
    monkeypatch.setattr(config, "BATCH_SIZE", 8)
