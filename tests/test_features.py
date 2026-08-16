"""Feature extraction: stream shapes, fused length, and cache correctness.

The fused length of 2376 is the number the base paper reports; if it drifts,
the comparison is no longer like-for-like.
"""
import os

import numpy as np
import pytest

import config
from features import (_items_fingerprint, build_feature_matrix, extract_streams,
                      fix_length, load_waveform)


def test_extract_streams_returns_expected_shapes(synth_wave):
    mfcc, zcr, rmse = extract_streams(synth_wave)
    assert mfcc.shape == (config.MFCC_LEN,)
    assert zcr.shape == (config.ZCR_LEN,)
    assert rmse.shape == (config.RMSE_LEN,)


def test_streams_concatenate_to_the_base_paper_input_length(synth_wave):
    mfcc, zcr, rmse = extract_streams(synth_wave)
    assert len(mfcc) + len(zcr) + len(rmse) == 2376


def test_streams_are_float32_and_finite(synth_wave):
    for arr in extract_streams(synth_wave):
        assert arr.dtype == np.float32
        assert np.all(np.isfinite(arr))


def test_fix_length_pads_short_waveforms():
    short = np.ones(1000, dtype=np.float32)
    out = fix_length(short)
    assert out.shape == (config.N_SAMPLES,)
    assert np.all(out[1000:] == 0)


def test_fix_length_truncates_long_waveforms():
    long = np.ones(config.N_SAMPLES * 2, dtype=np.float32)
    assert fix_length(long).shape == (config.N_SAMPLES,)


def test_load_waveform_normalises_and_fixes_length(synth_corpus):
    path = str(next(synth_corpus.rglob("*.wav")))
    y = load_waveform(path)
    assert y.shape == (config.N_SAMPLES,)
    assert np.max(np.abs(y)) <= 1.0 + 1e-6


def _items(paths):
    return [{"path": p, "emotion": "angry", "augment": False,
             "emotion_aware": False, "seed": 0} for p in paths]


def test_fingerprint_is_stable_for_identical_inputs(synth_corpus):
    items = _items([str(p) for p in synth_corpus.rglob("*.wav")][:5])
    assert _items_fingerprint(items) == _items_fingerprint(items)


def test_fingerprint_changes_with_feature_settings(synth_corpus, monkeypatch):
    items = _items([str(p) for p in synth_corpus.rglob("*.wav")][:5])
    before = _items_fingerprint(items)
    monkeypatch.setattr(config, "N_MFCC", 40)
    assert _items_fingerprint(items) != before


def test_fingerprint_changes_with_augmentation_seed(synth_corpus):
    paths = [str(p) for p in synth_corpus.rglob("*.wav")][:5]
    a = _items(paths)
    b = _items(paths)
    b[0] = dict(b[0], augment=True, seed=99)
    assert _items_fingerprint(a) != _items_fingerprint(b)


def test_build_feature_matrix_shapes_and_caching(synth_corpus, tmp_path,
                                                 monkeypatch, capsys):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    paths = [str(p) for p in synth_corpus.rglob("*.wav")][:6]
    items = _items(paths)

    out = build_feature_matrix(items, desc="unit")
    assert out["mfcc"].shape == (6, config.MFCC_LEN)
    assert out["zcr"].shape == (6, config.ZCR_LEN)
    assert out["rmse"].shape == (6, config.RMSE_LEN)
    assert out["y"].shape == (6,)

    capsys.readouterr()
    again = build_feature_matrix(items, desc="unit")
    assert "cache hit" in capsys.readouterr().out
    assert np.array_equal(out["mfcc"], again["mfcc"])


def test_augmented_items_produce_correct_shapes(synth_corpus, tmp_path,
                                                monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache2"))
    paths = [str(p) for p in synth_corpus.rglob("*.wav")][:4]
    items = [{"path": p, "emotion": "sad", "augment": True,
              "emotion_aware": True, "seed": 11 + i}
             for i, p in enumerate(paths)]
    out = build_feature_matrix(items, desc="aug")
    assert out["mfcc"].shape == (4, config.MFCC_LEN)
