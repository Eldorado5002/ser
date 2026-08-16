"""Filename -> emotion label parsing for all four corpora.

Label parsing depends only on filenames, never on audio, so these tests are
exhaustive and free. They are the first line of defence: a parser that returns
None does not crash - it silently drops samples.
"""
import pytest

import config
from data_loader import (_parse_cremad, _parse_ravdess, _parse_savee,
                         _parse_tess)


@pytest.mark.parametrize("code,expected", [
    ("01", "neutral"),
    ("02", "neutral"),   # calm merged into neutral (7-class setup)
    ("03", "happy"),
    ("04", "sad"),
    ("05", "angry"),
    ("06", "fear"),
    ("07", "disgust"),
    ("08", "surprise"),
])
def test_ravdess_emotion_codes(code, expected):
    assert _parse_ravdess(f"03-01-{code}-01-02-01-12.wav") == expected


def test_ravdess_handles_full_paths():
    path = "/kaggle/input/x/audio_speech_actors_01-24/Actor_06/03-01-05-01-01-01-06.wav"
    assert _parse_ravdess(path) == "angry"


def test_ravdess_rejects_malformed():
    assert _parse_ravdess("03-01.wav") is None


@pytest.mark.parametrize("token,expected", [
    ("angry", "angry"),
    ("disgust", "disgust"),
    ("fear", "fear"),
    ("happy", "happy"),
    ("neutral", "neutral"),
    ("sad", "sad"),
    ("ps", "surprise"),          # "pleasant surprise"
    ("surprise", "surprise"),
])
def test_tess_emotion_tokens(token, expected):
    assert _parse_tess(f"OAF_back_{token}.wav") == expected


def test_tess_is_case_insensitive():
    assert _parse_tess("YAF_dog_FEAR.wav") == "fear"
    assert _parse_tess("OAF_back_PS.wav") == "surprise"


@pytest.mark.parametrize("code,expected", [
    ("a01", "angry"),
    ("d03", "disgust"),
    ("f11", "fear"),
    ("h07", "happy"),
    ("n22", "neutral"),
    ("sa15", "sad"),        # two-letter code must win over "s"
    ("su09", "surprise"),   # two-letter code must win over "s"
])
def test_savee_emotion_codes(code, expected):
    assert _parse_savee(f"DC_{code}.wav") == expected


def test_savee_two_letter_codes_take_precedence():
    """'sa' and 'su' must not be truncated to a single letter."""
    assert _parse_savee("JE_sa01.wav") == "sad"
    assert _parse_savee("KL_su01.wav") == "surprise"
    assert _parse_savee("JE_sa01.wav") != _parse_savee("KL_su01.wav")


def test_savee_without_speaker_prefix():
    assert _parse_savee("a01.wav") == "angry"
    assert _parse_savee("sa01.wav") == "sad"


@pytest.mark.parametrize("token,expected", [
    ("ANG", "angry"),
    ("DIS", "disgust"),
    ("FEA", "fear"),
    ("HAP", "happy"),
    ("NEU", "neutral"),
    ("SAD", "sad"),
])
def test_cremad_emotion_tokens(token, expected):
    assert _parse_cremad(f"1091_DFA_{token}_XX.wav") == expected


def test_cremad_has_no_surprise_class():
    """CREMA-D contains no surprise recordings; nothing may map to it."""
    from data_loader import _CREMAD_MAP
    assert "surprise" not in _CREMAD_MAP.values()


def test_cremad_rejects_malformed():
    assert _parse_cremad("1091_DFA.wav") is None


def test_every_parser_output_is_a_known_emotion():
    samples = [
        (_parse_ravdess, "03-01-06-01-02-01-12.wav"),
        (_parse_tess, "OAF_back_ps.wav"),
        (_parse_savee, "DC_su01.wav"),
        (_parse_cremad, "1091_DFA_ANG_XX.wav"),
    ]
    for parser, name in samples:
        assert parser(name) in config.EMOTION_TO_ID
