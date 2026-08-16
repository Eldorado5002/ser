"""Aggregate finished runs into report-ready tables and an acceptance check.

Usage:
    python report.py                    # reads runs/, writes report/
    python report.py --runs-dir runs --out-dir report
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

import config

#: Accuracy reported by the base paper (Chourasia et al., 2026).
BASE_PAPER_ACCURACY = 0.9491
BASE_PAPER_MACRO_F1 = 0.94

#: The six configurations that must be present for a complete sweep.
REQUIRED_TAGS = ["base", "afw", "eaaa", "mstc", "cadl", "full"]

_PAIR_KEYS = ["sad<->neutral", "angry<->fear"]


def collect_runs(runs_dir: str = config.RUNS_DIR) -> pd.DataFrame:
    """Load every runs/<tag>/test_metrics.json into one flat DataFrame."""
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*",
                                              "test_metrics.json"))):
        with open(path) as f:
            m = json.load(f)
        cfg = m.get("config", {})
        row = {
            "tag": os.path.basename(os.path.dirname(path)),
            "accuracy": m.get("accuracy"),
            "ci_low": (m.get("accuracy_95ci") or [None, None])[0],
            "ci_high": (m.get("accuracy_95ci") or [None, None])[1],
            "macro_f1": m.get("macro_f1"),
            "macro_precision": m.get("macro_precision"),
            "macro_recall": m.get("macro_recall"),
            "macro_specificity": m.get("macro_specificity"),
            "macro_gmean": m.get("macro_gmean"),
            "mcc": m.get("mcc"),
            "cohen_kappa": m.get("cohen_kappa"),
            "auc": m.get("auc_ovr_macro"),
            "n_test": m.get("n_test"),
            "use_afw": cfg.get("use_afw"),
            "use_eaaa": cfg.get("use_eaaa"),
            "use_mstc": cfg.get("use_mstc"),
            "use_cadl": cfg.get("use_cadl"),
            "params": cfg.get("params"),
        }
        for key in _PAIR_KEYS:
            row[key] = (m.get("confusion_pair_errors") or {}).get(key)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def check_acceptance(df: pd.DataFrame) -> list:
    """Evaluate the spec's acceptance criteria. Returns (name, passed, detail)."""
    results = []

    def add(name, ok, detail):
        results.append((name, bool(ok), detail))

    if df.empty:
        add("Runs present", False, "no runs found")
        return results

    by_tag = df.set_index("tag")
    present = set(by_tag.index)

    missing = [t for t in REQUIRED_TAGS if t not in present]
    add("All six configurations present", not missing,
        "missing: " + ", ".join(missing) if missing else
        "base, afw, eaaa, mstc, cadl, full")

    if "base" in present:
        acc = by_tag.loc["base", "accuracy"]
        add("Base reproduces the base paper (>=0.93)", acc >= 0.93,
            f"base accuracy = {acc:.4f} (paper: {BASE_PAPER_ACCURACY:.4f})")

    if "full" in present:
        acc = by_tag.loc["full", "accuracy"]
        f1 = by_tag.loc["full", "macro_f1"]
        add("Full beats the base paper", acc > BASE_PAPER_ACCURACY,
            f"full accuracy = {acc:.4f} vs {BASE_PAPER_ACCURACY:.4f}")
        add("Full hits the 95.5-96.5% target", 0.955 <= acc <= 0.965,
            f"full accuracy = {acc:.4f}")
        add("Full macro F1 > 0.94", f1 > BASE_PAPER_MACRO_F1,
            f"macro F1 = {f1:.4f}")

    if {"base", "cadl"} <= present:
        base_err = sum(by_tag.loc["base", k] or 0 for k in _PAIR_KEYS)
        cadl_err = sum(by_tag.loc["cadl", k] or 0 for k in _PAIR_KEYS)
        add("CADL reduces confusion-pair errors", cadl_err < base_err,
            f"base = {base_err}, +CADL = {cadl_err}")

    if by_tag["params"].notna().any():
        p = by_tag["params"].dropna()
        add("Parameter counts stay in the 6.5-8.5 M band",
            bool(((p > 6.5e6) & (p < 8.5e6)).all()),
            f"min = {int(p.min()):,}, max = {int(p.max()):,}")

    if {"base", "afw"} <= present:
        extra = by_tag.loc["afw", "params"] - by_tag.loc["base", "params"]
        add("AFW adds <0.1% parameters",
            0 < extra < 0.001 * by_tag.loc["base", "params"],
            f"+{int(extra):,} parameters")

    return results


def _ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["tag", "use_afw", "use_eaaa", "use_mstc", "use_cadl", "accuracy",
            "macro_f1", "mcc", "cohen_kappa", "auc"] + _PAIR_KEYS + ["params"]
    order = {t: i for i, t in enumerate(REQUIRED_TAGS)}
    out = df[[c for c in cols if c in df.columns]].copy()
    out["_o"] = out["tag"].map(lambda t: order.get(t, 99))
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def write_report(runs_dir: str = config.RUNS_DIR,
                 out_dir: str = "report") -> pd.DataFrame:
    """Write summary.csv, ablation_table.csv, acceptance.md and REPORT.md."""
    os.makedirs(out_dir, exist_ok=True)
    df = collect_runs(runs_dir)

    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    table = _ablation_table(df) if not df.empty else pd.DataFrame()
    table.to_csv(os.path.join(out_dir, "ablation_table.csv"), index=False)

    checks = check_acceptance(df)
    lines = ["# Acceptance Criteria", ""]
    for name, ok, detail in checks:
        lines.append(f"- [{'x' if ok else ' '}] **{name}** - {detail}")
    with open(os.path.join(out_dir, "acceptance.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    md = ["# SER Results", "",
          "## Headline comparison", "",
          "| Model | Accuracy | Macro F1 |", "|---|---|---|",
          f"| Base paper (Chourasia et al., 2026) | "
          f"{BASE_PAPER_ACCURACY * 100:.2f}% | {BASE_PAPER_MACRO_F1:.2f} |"]

    if not df.empty:
        by_tag = df.set_index("tag")
        for tag, label in (("base", "This work - base reproduction"),
                           ("full", "This work - proposed (all 4 novelties)")):
            if tag in by_tag.index:
                md.append(f"| {label} | "
                          f"{by_tag.loc[tag, 'accuracy'] * 100:.2f}% | "
                          f"{by_tag.loc[tag, 'macro_f1']:.4f} |")

    md += ["", "## Ablation study", "",
           table.to_markdown(index=False) if not table.empty
           else "_no runs found_", "",
           "## Acceptance", ""] + lines[2:]

    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default=config.RUNS_DIR)
    p.add_argument("--out-dir", default="report")
    args = p.parse_args()

    df = write_report(args.runs_dir, args.out_dir)
    if df.empty:
        print(f"[report] no runs found under {args.runs_dir}/")
        return

    print(_ablation_table(df).to_string(index=False))
    print()
    for name, ok, detail in check_acceptance(df):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n[report] written to {args.out_dir}/")


if __name__ == "__main__":
    main()
