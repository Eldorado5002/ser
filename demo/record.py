"""Record from the microphone and predict the emotion, live.

    python demo/record.py                 # one 4-second take
    python demo/record.py --loop          # keep going until Ctrl+C
    python demo/record.py --seconds 5

Note on expectations: the pipeline uses a 0.6 s offset and a 2.5 s window, so
speak from the moment recording starts and keep going for the whole take.

This is the honest test. The trained model never heard your voice, whereas the
TESS demo clips come from a speaker that appears in the training split - so
expect noticeably lower confidence here. Measured test accuracy on the full
corpus is 57.95%.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ser_demo import BOLD, DIM, RESET, featurise, load, render  # noqa: E402

import config  # noqa: E402


def countdown(n: int = 3) -> None:
    for i in range(n, 0, -1):
        print(f"\r  {DIM}starting in{RESET} {BOLD}{i}{RESET} ...", end="",
              flush=True)
        time.sleep(0.7)
    print("\r" + " " * 40, end="\r")


def capture(seconds: float) -> np.ndarray:
    import sounddevice as sd

    sr = config.SAMPLE_RATE
    print(f"  {BOLD}\033[91m● RECORDING{RESET} - speak now "
          f"({seconds:.1f}s) ...", end="", flush=True)
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1,
                   dtype="float32")
    sd.wait()
    print(f"\r  {DIM}captured {seconds:.1f}s{RESET}" + " " * 30)
    return audio.reshape(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--model", default="full", choices=["full", "mstc"])
    ap.add_argument("--loop", action="store_true",
                    help="keep recording until Ctrl+C")
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="also write each take to DIR as a .wav")
    args = ap.parse_args()

    print(f"{DIM}loading {args.model} model ...{RESET}")
    model, weight_model, scalers = load(args.model)
    print(f"{DIM}{model.count_params():,} parameters — ready{RESET}")

    take = 0
    try:
        while True:
            take += 1
            print()
            input(f"  {BOLD}press Enter to record take {take}{RESET} "
                  f"(Ctrl+C to quit) ")
            countdown()
            wave = capture(args.seconds)

            peak = float(np.max(np.abs(wave)))
            if peak < 0.01:
                print(f"  {DIM}almost silent (peak {peak:.4f}) - is the "
                      f"microphone muted?{RESET}")
                if not args.loop:
                    return
                continue
            wave = wave / peak

            if args.save:
                import soundfile as sf
                os.makedirs(args.save, exist_ok=True)
                p = os.path.join(args.save, f"take_{take:02d}.wav")
                sf.write(p, wave, config.SAMPLE_RATE)
                print(f"  {DIM}saved {p}{RESET}")

            from features import fix_length
            x = featurise(fix_length(wave), scalers)
            probs = model.predict(x, verbose=0)[0]
            w = weight_model.predict(x, verbose=0)[0] if weight_model else None

            render(probs, weights=w, truth=None, title=f"take {take}")
            print(f"  {DIM}{'-' * 66}{RESET}")

            if not args.loop:
                return
    except (KeyboardInterrupt, EOFError):
        print(f"\n  {DIM}done{RESET}\n")


if __name__ == "__main__":
    main()
