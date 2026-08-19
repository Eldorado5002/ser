"""Predict the emotion of one or more .wav files.

    python demo/predict.py demo/clips/OAF_back_angry.wav
    python demo/predict.py demo/clips              # every clip in a folder
    python demo/predict.py demo/clips --model mstc

Prints the full probability distribution rather than a single label: the
model is ~58% accurate on this corpus, so the distribution is the honest
output and the runner-up is often informative.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ser_demo import (BOLD, DIM, RESET, featurise, label_from_name, load,  # noqa
                      load_clip, render)


def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.wav"))))
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help=".wav files or a directory")
    ap.add_argument("--model", default="best",
                    choices=["best", "full", "mstc"],
                    help="best = the report's headline configuration "
                         "(60.79%%, 2.54 M params); full = all four novelties, "
                         "the only one that shows AFW stream weights; "
                         "mstc = best single novelty")
    args = ap.parse_args()

    files = collect(args.paths)
    if not files:
        raise SystemExit("no .wav files found")

    from ser_demo import MODEL_INFO
    print(f"{DIM}loading '{args.model}' - {MODEL_INFO[args.model]}{RESET}")
    model, weight_model, scalers = load(args.model)
    print(f"{DIM}{model.count_params():,} parameters{RESET}")

    hits = graded = 0
    for path in files:
        wave = load_clip(path)
        x = featurise(wave, scalers)
        probs = model.predict(x, verbose=0)[0]
        w = weight_model.predict(x, verbose=0)[0] if weight_model else None
        truth = label_from_name(path)

        render(probs, weights=w, truth=truth,
               title=f"{os.path.basename(path)}")
        if truth is not None:
            graded += 1
            hits += int(config_top(probs) == truth)
        print(f"  {DIM}{'-' * 66}{RESET}")

    if graded > 1:
        print(f"\n{BOLD}  {hits}/{graded} correct "
              f"({100*hits/graded:.0f}%) on this sample{RESET}")
        print(f"  {DIM}Measured test-set accuracy is 57.95% (n=2,433); a "
              f"handful of clips will vary widely around that.{RESET}\n")


def config_top(probs):
    import config
    return config.EMOTIONS[int(np.argmax(probs))]


if __name__ == "__main__":
    main()
