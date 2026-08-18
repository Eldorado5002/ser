# Regularisation sweep — results

**Run:** Kaggle kernel `macbot000/notebook146f8a3bc7`, 2026-08-18 23:07 → 74 min, GPU P100
**Protocol:** six candidates, novelties off, scored on **validation only**. The test set was never loaded.

---

## Results

| Candidate | Val acc | Train acc | Gap | Params | Best epoch | Δ vs base |
|---|---|---|---|---|---|---|
| **`gap_reg_aug3`** | **60.84%** | 62.50% | 1.7 | **2,540,167** | 5 | **+2.26** |
| `gap_reg` | 60.33% | 64.51% | 4.2 | 2,540,167 | 9 | +1.75 |
| `reg` | 59.40% | 60.13% | 0.7 | 7,324,295 | 6 | +0.82 |
| `gap` | 58.79% | 58.44% | −0.3 | 2,540,167 | 4 | +0.21 |
| `base` | 58.58% | 63.32% | 4.7 | 7,324,295 | 4 | — |
| `gap_reg_aug3_lr` | 58.58% | 63.43% | 4.8 | 2,540,167 | 5 | +0.00 |

**Winner on validation: `gap_reg_aug3`** — GlobalAveragePooling head + dropout 0.35/0.55 + L2 1e-4 + 3× augmentation.

---

## Honest assessment

**The sweep helped, but far less than predicted.** The estimate given beforehand was 65–72%; the actual gain was **+2.26 points** (58.58% → 60.84%). That estimate was too optimistic and should not be repeated in the report.

### Why — a correction to the earlier diagnosis

The earlier claim was that the model is "severely under-constrained," citing a **37-point** train/validation gap (98.9% vs 61.8%). That number is real but was measured at **epoch 14**. Training uses `EarlyStopping(restore_best_weights=True)` on validation loss, which restores **epoch 4** — where the gap is only **4.7 points**.

So the *evaluated* model was never badly overfit. Early stopping was already doing most of the regularisation work, which is precisely why adding more of it yielded so little. The diagnosis identified a real phenomenon but attributed the wrong cause to the accuracy ceiling.

The genuine limitation appears to be representational: validation accuracy plateaus at 59–61% across every configuration tried, including one with 65% fewer parameters and one with 3× the data. That points at the **features** (20 MFCC + ZCR + RMSE) rather than at model capacity or regularisation.

### What the sweep did establish

**1. A real efficiency result.** `gap` alone matches base accuracy (58.79% vs 58.58%) with **65% fewer parameters** — 2.54 M against 7.32 M. The `Flatten(9,472) → Dense(512)` head holds 4.85 M parameters and contributes nothing. This directly supports the project's lightweight/edge-deployment thesis and is worth reporting in its own right.

**2. The best configuration is both better and smaller:** +2.26 points accuracy at 35% of the parameter count.

**3. The learning-rate hypothesis was wrong.** `gap_reg_aug3_lr` (lr 5e-4) tied with base at 58.58%, undoing the gain from `gap_reg_aug3`. Lowering the learning rate did not help.

---

## An unexploited win worth ~3 points

Inspecting the base run's per-epoch history:

| Epoch | 4 | 9 | 11 | 12 | **13** | 14 |
|---|---|---|---|---|---|---|
| val accuracy | 58.58% | 59.71% | 61.05% | 60.84% | **61.87%** | 61.77% |
| val loss | **1.1136** | 1.3231 | 1.4858 | 1.5066 | 1.5056 | 1.6032 |

Validation **loss** is best at epoch 4. Validation **accuracy** peaks at epoch 13 — **3.29 points higher**.

The two diverge because cross-entropy loss punishes growing over-confidence even while the argmax keeps improving. Since `EarlyStopping` and `ModelCheckpoint` both monitor `val_loss`, the pipeline discards the more accurate model.

**Switching the monitor to `val_accuracy` is a one-line change and is worth roughly 3 points** — more than the entire regularisation sweep delivered. Combined with `gap_reg_aug3`, a plausible figure is **63–65%**.

Caveat: `val_loss` is the more conservative criterion, and selecting on accuracy is slightly more prone to noise. Either is defensible provided the choice is stated.

---

## Status

- Test set: **still sealed**. No test evaluation has been run on any sweep candidate.
- All six models saved as `sweep_<tag>.keras` in the kernel output.
- Raw results: `results/sweep/sweep_results.json`.

## Next step

Run **`notebooks/07_final_evaluation.ipynb`** on Kaggle. It selects the winner by validation accuracy, evaluates it and the `base` control on the test set exactly once, and produces the per-corpus breakdown. Roughly 5 minutes; inference only.
