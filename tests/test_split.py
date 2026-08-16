"""Train/validation/test split integrity.

The split must happen BEFORE augmentation (spec B.4.6) and must never place
the same recording on both sides of the boundary.
"""
import pandas as pd

import config
from data_loader import build_metadata, split_metadata


def test_split_ratios_are_72_8_20(corpus_dirs):
    df = build_metadata(verbose=False, strict=False)
    train, val, test = split_metadata(df)

    n = len(df)
    assert len(train) + len(val) + len(test) == n
    assert abs(len(test) / n - 0.20) < 0.03
    assert abs(len(val) / n - 0.08) < 0.03
    assert abs(len(train) / n - 0.72) < 0.03


def test_no_path_overlap_between_splits(corpus_dirs):
    df = build_metadata(verbose=False, strict=False)
    train, val, test = split_metadata(df)

    tr, va, te = set(train.path), set(val.path), set(test.path)
    assert tr & te == set()
    assert tr & va == set()
    assert va & te == set()


def test_every_class_appears_in_every_split(corpus_dirs):
    df = build_metadata(verbose=False, strict=False)
    train, val, test = split_metadata(df)

    for name, part in (("train", train), ("test", test)):
        assert set(part.emotion) == set(df.emotion), name


def test_split_is_stratified(corpus_dirs):
    """Class proportions must be preserved in the test split."""
    df = build_metadata(verbose=False, strict=False)
    _, _, test = split_metadata(df)

    overall = df.emotion.value_counts(normalize=True)
    in_test = test.emotion.value_counts(normalize=True)
    for emotion in overall.index:
        assert abs(overall[emotion] - in_test.get(emotion, 0)) < 0.10, emotion


def test_split_is_deterministic(corpus_dirs):
    """Same seed must give the same partition, or cached features desync."""
    df = build_metadata(verbose=False, strict=False)
    a1, b1, c1 = split_metadata(df)
    a2, b2, c2 = split_metadata(df)

    assert list(a1.path) == list(a2.path)
    assert list(b1.path) == list(b2.path)
    assert list(c1.path) == list(c2.path)
