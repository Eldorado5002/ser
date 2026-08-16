"""Config arithmetic and class-ordering invariants.

These constants are load-bearing: INPUT_LEN feeds the model's Reshape layer,
and the alphabetical class order determines CADL's confusion-pair indices.
"""
import config


def test_input_length_is_exactly_2376():
    assert config.INPUT_LEN == 2376


def test_input_length_is_the_sum_of_its_streams():
    assert config.MFCC_LEN + config.ZCR_LEN + config.RMSE_LEN == config.INPUT_LEN


def test_frame_count_matches_librosa_centred_framing():
    assert config.N_SAMPLES == 55125
    assert config.N_FRAMES == 108
    assert config.MFCC_LEN == config.N_MFCC * config.N_FRAMES


def test_emotions_are_alphabetical():
    assert config.EMOTIONS == sorted(config.EMOTIONS)
    assert len(config.EMOTIONS) == config.NUM_CLASSES == 7


def test_confusion_pairs_resolve_to_expected_indices():
    pairs = [(config.EMOTION_TO_ID[a], config.EMOTION_TO_ID[b])
             for a, b in config.CONFUSION_PAIRS]
    assert pairs == [(5, 4), (0, 2)]  # sad<->neutral, angry<->fear


def test_id_mapping_round_trips():
    for i, e in enumerate(config.EMOTIONS):
        assert config.EMOTION_TO_ID[e] == i
        assert config.ID_TO_EMOTION[i] == e
