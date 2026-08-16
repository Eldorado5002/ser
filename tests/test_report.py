"""Aggregation of finished runs into report-ready tables."""
import json
import os

import pandas as pd
import pytest

import config
from report import check_acceptance, collect_runs, write_report

TAGS = ["base", "afw", "eaaa", "mstc", "cadl", "full"]


def _fake_metrics(acc, pair_errors, params=7_190_000, **flags):
    return {
        "accuracy": acc,
        "accuracy_95ci": [acc - 0.01, acc + 0.01],
        "macro_precision": acc - 0.01, "macro_recall": acc - 0.01,
        "macro_f1": acc - 0.005, "weighted_f1": acc,
        "specificity_per_class": {e: 0.99 for e in config.EMOTIONS},
        "macro_specificity": 0.99,
        "gmean_per_class": {e: acc for e in config.EMOTIONS},
        "macro_gmean": acc, "mcc": acc - 0.02, "cohen_kappa": acc - 0.02,
        "auc_ovr_macro": 0.996,
        "confusion_pair_errors": pair_errors,
        "n_test": 2433,
        "config": {"use_afw": flags.get("afw", False),
                   "use_eaaa": flags.get("eaaa", False),
                   "use_mstc": flags.get("mstc", False),
                   "use_cadl": flags.get("cadl", False),
                   "params": params},
    }


@pytest.fixture
def fake_runs(tmp_path):
    specs = {
        "base": (0.9491, {"sad<->neutral": 60, "angry<->fear": 40}, {},
                 7_190_000),
        "afw":  (0.9530, {"sad<->neutral": 58, "angry<->fear": 39},
                 {"afw": True}, 7_193_000),
        "eaaa": (0.9515, {"sad<->neutral": 59, "angry<->fear": 41},
                 {"eaaa": True}, 7_190_000),
        "mstc": (0.9540, {"sad<->neutral": 57, "angry<->fear": 38},
                 {"mstc": True}, 7_189_996),
        "cadl": (0.9522, {"sad<->neutral": 44, "angry<->fear": 30},
                 {"cadl": True}, 7_190_000),
        "full": (0.9601, {"sad<->neutral": 39, "angry<->fear": 26},
                 {"afw": True, "eaaa": True, "mstc": True, "cadl": True},
                 7_192_996),
    }
    for tag, (acc, pairs, flags, params) in specs.items():
        d = tmp_path / tag
        d.mkdir(parents=True)
        with open(d / "test_metrics.json", "w") as f:
            json.dump(_fake_metrics(acc, pairs, params=params, **flags), f)
    return tmp_path


def test_collect_runs_finds_every_run(fake_runs):
    df = collect_runs(str(fake_runs))
    assert len(df) == 6
    assert set(df["tag"]) == set(TAGS)


def test_collect_runs_flattens_the_key_metrics(fake_runs):
    df = collect_runs(str(fake_runs)).set_index("tag")
    assert df.loc["full", "accuracy"] == pytest.approx(0.9601)
    assert df.loc["base", "sad<->neutral"] == 60
    assert df.loc["cadl", "angry<->fear"] == 30
    assert df.loc["base", "params"] == 7_190_000


def test_collect_runs_returns_empty_frame_when_nothing_exists(tmp_path):
    assert collect_runs(str(tmp_path)).empty


def test_collect_runs_ignores_directories_without_metrics(fake_runs):
    (fake_runs / "half_finished").mkdir()
    assert len(collect_runs(str(fake_runs))) == 6


def test_acceptance_passes_on_a_good_sweep(fake_runs):
    results = check_acceptance(collect_runs(str(fake_runs)))
    failed = [(n, d) for n, ok, d in results if not ok]
    assert failed == []


def test_acceptance_flags_a_missing_configuration(fake_runs):
    df = collect_runs(str(fake_runs))
    results = check_acceptance(df[df.tag != "mstc"])
    assert any("mstc" in d and not ok for _, ok, d in results)


def test_acceptance_flags_cadl_not_reducing_confusion(fake_runs):
    df = collect_runs(str(fake_runs))
    df.loc[df.tag == "cadl", "sad<->neutral"] = 99
    df.loc[df.tag == "cadl", "angry<->fear"] = 99
    results = check_acceptance(df)
    assert any("confusion" in n.lower() and not ok for n, ok, _ in results)


def test_write_report_produces_all_outputs(fake_runs, tmp_path):
    out = tmp_path / "report"
    write_report(str(fake_runs), str(out))
    for name in ("summary.csv", "ablation_table.csv", "acceptance.md",
                 "REPORT.md"):
        assert (out / name).exists(), name


def test_report_markdown_contains_the_headline_comparison(fake_runs, tmp_path):
    out = tmp_path / "report"
    write_report(str(fake_runs), str(out))
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "94.91" in text        # base paper reference
    assert "96.01" in text        # our full result
