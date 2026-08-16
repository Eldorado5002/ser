"""Duplicate-file detection.

Two Kaggle mirrors ship every file twice:
  RAVDESS - Actor_01..24/ AND audio_speech_actors_01-24/
  TESS    - "TESS Toronto..." AND "tess toronto..." (case-differing dirs)

Because duplicates are byte-identical recordings and the split is random over
utterances, a clip and its twin land on opposite sides of the train/test
boundary ~32% of the time. That contaminates ~41% of the test set with exact
training copies, and raises no error. Hence this guard.
"""
import json
import os

import pytest

import config
from data_loader import (CANONICAL_SUBDIRS, EXPECTED_COUNTS,
                         check_no_duplicate_basenames)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "kaggle_filelists.json")


@pytest.fixture(scope="module")
def real_listings():
    with open(FIXTURE) as f:
        return json.load(f)


def test_fixture_captures_the_upstream_defect(real_listings):
    """Guards the guard: if this fails, the fixture no longer reflects the bug."""
    assert len(real_listings["RAVDESS"]) == 2880   # 2x 1440
    assert len(real_listings["TESS"]) == 5600      # 2x 2800
    assert len(real_listings["SAVEE"]) == 480      # clean
    assert len(real_listings["CREMA-D"]) == 7442   # clean


def test_guard_raises_on_duplicated_ravdess(real_listings):
    with pytest.raises(ValueError, match="duplicate"):
        check_no_duplicate_basenames(real_listings["RAVDESS"], "RAVDESS")


def test_guard_raises_on_duplicated_tess(real_listings):
    with pytest.raises(ValueError, match="duplicate"):
        check_no_duplicate_basenames(real_listings["TESS"], "TESS")


def test_guard_accepts_clean_corpora(real_listings):
    check_no_duplicate_basenames(real_listings["SAVEE"], "SAVEE")
    check_no_duplicate_basenames(real_listings["CREMA-D"], "CREMA-D")


def test_error_message_names_the_corpus_and_a_fix(real_listings):
    with pytest.raises(ValueError) as exc:
        check_no_duplicate_basenames(real_listings["RAVDESS"], "RAVDESS")
    msg = str(exc.value)
    assert "RAVDESS" in msg
    assert "1440" in msg                        # how many duplicated
    assert CANONICAL_SUBDIRS["RAVDESS"] in msg  # how to fix it


def test_canonical_subdirs_deduplicate_every_corpus(real_listings):
    for corpus, subdir in CANONICAL_SUBDIRS.items():
        kept = [p for p in real_listings[corpus]
                if p.replace("\\", "/").startswith(subdir)]
        assert len(kept) == EXPECTED_COUNTS[corpus], corpus
        check_no_duplicate_basenames(kept, corpus)


def test_canonical_subdirs_sum_to_the_spec_target(real_listings):
    total = sum(
        len([p for p in real_listings[c]
             if p.replace("\\", "/").startswith(s)])
        for c, s in CANONICAL_SUBDIRS.items())
    assert total == 12162


def test_expected_counts_match_the_spec():
    assert EXPECTED_COUNTS == {"RAVDESS": 1440, "TESS": 2800,
                               "SAVEE": 480, "CREMA-D": 7442}
    assert sum(EXPECTED_COUNTS.values()) == 12162
