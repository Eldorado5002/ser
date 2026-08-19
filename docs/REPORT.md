# Adaptive Feature-Weighted 1D-CNN with Emotion-Aware Augmentation for Robust Speech Emotion Recognition

**Project report**  
Domain: Speech Processing & Affective Computing  
Base paper: Chourasia, N., Lamba, C. S., & Gupta, A. K. (2026). *A 1D-CNN with advanced data augmentation for robust speech emotion recognition.* Scientific Reports.

---

## Abstract

We implement a lightweight Conv1D speech-emotion-recognition framework extending a published 1D-CNN baseline with four novelties: adaptive feature weighting (AFW), emotion-aware adaptive augmentation (EAAA), multi-scale temporal convolution (MSTC), and a confusion-aware discriminative loss (CADL). All four are evaluated by a component-wise ablation on a fused four-corpus dataset (RAVDESS, TESS, SAVEE, CREMA-D; 12,162 utterances, seven emotions) — the ablation the base paper listed as missing future work.

On a correctly constructed evaluation protocol the base configuration reaches **57.95%** test accuracy, against the **94.91%** reported by the base paper. Investigating that gap became the project's principal finding. We identify and quantify two data-leakage mechanisms that inflate accuracy on this corpus: duplicated dataset mirrors (**+14.71** points) and augmenting before splitting (**+24.27** points). Each was predicted in advance and confirmed by direct measurement of the contamination rate. Our honest figure exceeds the closest comparable published multi-corpus result (50.6%, Dasude et al., 2024) by 7.4 points.

## 1. Contributions

1. **A complete, tested implementation** of the base architecture plus four novelties, with a 117-test suite verifying every claim the report makes (Section 4).
2. **The component-wise ablation** the base paper identified as missing future work (Section 6).
3. **A reproducibility investigation** that quantifies two leakage mechanisms responsible for inflated multi-corpus SER accuracies (Section 7) — the principal finding.
4. **An honest multi-corpus baseline** that exceeds the closest comparable published result on the same four corpora.

## 2. Method

The backbone follows the base paper: three frame-level acoustic streams (20 MFCC, ZCR, RMSE) fused into a 2,376-dimensional sequential input, then five Conv1D stages (512-512-256-256-128) with batch normalisation and max pooling, a Dense(512) head, and a 7-way softmax. Total parameters: **7,324,295**.

| # | Novelty | Addresses | Acts at | Cost |
|---|---|---|---|---|
| N1 | **AFW** — Adaptive Feature Weighting | equal treatment of the three streams | feature fusion | +163 params |
| N2 | **EAAA** — Emotion-Aware Adaptive Augmentation | uniform augmentation | data (training only) | zero |
| N3 | **MSTC** — Multi-Scale Temporal Convolution | single-scale kernels | first conv stage | -4 params |
| N4 | **CADL** — Confusion-Aware Discriminative Loss | confusion-blind objective | training objective | zero at inference |

AFW summarises each stream by its mean and standard deviation, passes them through a compact dense layer and a softmax, and rescales each stream by three times its weight — so the equal-importance solution (1/3, 1/3, 1/3) is the identity, making the base-versus-AFW comparison clean. MSTC replaces the first conv stage with parallel kernels of size 3, 5 and 7, splitting the 512-filter budget 172/170/170 so the parameter count is preserved. CADL is focal cross-entropy (gamma=2) plus a pairwise penalty (lambda=0.5) on the sad-neutral and angry-fear pairs.

## 3. Dataset

Four corpora fused into one seven-class dataset, split 72:8:20 **before** any augmentation.

| Corpus | Utterances | Notes |
|---|---|---|
| RAVDESS | 1,440 | `calm` merged into `neutral` |
| TESS | 2,800 | 2 female speakers |
| SAVEE | 480 | 4 male speakers |
| CREMA-D | 7,442 | 91 actors; **no surprise class** |
| **Total** | **12,162** | |

| Emotion | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| Count | 1,923 | 1,923 | 1,923 | 1,923 | 1,895 | 1,923 | **652** |

Split sizes: **train 8,756 / validation 973 / test 2,433**, matching the base paper's protocol exactly.

> **Data-integrity note.** The RAVDESS and TESS Kaggle mirrors each ship the corpus twice (RAVDESS as both `Actor_01..24/` and `audio_speech_actors_01-24/`; TESS as two directories differing only in capitalisation). A naive recursive scan yields **16,402** files instead of 12,162. Section 7 shows why this matters.

## 4. Implementation and verification

Because the central claims are quantitative, each is backed by an automated test. The suite runs in ~70 seconds and requires **no audio download** — label parsing depends only on filenames, and the remaining tests use synthetic waveforms.

| Test module | Verifies |
|---|---|
| `test_config.py` | fused input length is exactly 2,376; class order and confusion-pair indices |
| `test_data_loader.py` | all four filename conventions, including SAVEE's two-letter `sa`/`su` precedence and RAVDESS `calm`→`neutral` |
| `test_no_duplicates.py` | the duplicate guard fires on the real 16,402-path listings; canonical subdirectories sum to exactly 12,162 |
| `test_split.py` | 72:8:20 ratios, stratification, **zero** train/test path overlap |
| `test_augmentation.py` | EAAA applies the per-class policy; reproducibility; length restored after time-stretch |
| `test_features.py` | stream shapes; cache fingerprint invalidation |
| `test_model.py` | AFW weights form a probability distribution and the uniform solution is the identity; **MSTC is parameter-neutral** |
| `test_losses.py` | **CADL reduces exactly to categorical cross-entropy** when both terms are disabled |
| `test_train_integration.py` | the saved checkpoint reproduces the reported metrics |

**117 tests pass** on both TensorFlow 2.15/Keras 2 and TensorFlow 2.20/Keras 3.

Two equivalence tests deserve emphasis: *CADL ≡ cross-entropy when disabled* and *MSTC is parameter-neutral*. Without them, a difference between `base` and `+CADL` could be a loss-scaling artefact, and a difference between `base` and `+MSTC` could be extra capacity. They are what make the ablation interpretable.

Two defects were found and fixed during verification:

- **Checkpoint/metrics mismatch.** `ModelCheckpoint` monitored `val_accuracy` while `EarlyStopping(restore_best_weights=True)` monitored `val_loss`. Aligning them was insufficient: Keras only restores best weights when early stopping actually *fires*, so a run completing all epochs reported final-epoch metrics while the saved file held a different epoch. The pipeline now reloads the checkpoint explicitly before evaluation.
- **Duplicate-mirror contamination**, described in Section 7.

## 5. Experimental setup

Adam (lr 1e-3), batch size 32, up to 50 epochs, early stopping on validation loss (patience 10), ReduceLROnPlateau. Per-stream standardisation fitted on the training partition only. Augmentation expands the training set to 11,700 samples for every configuration, so training-set size is never a confound. Six configurations were run on an NVIDIA P100. Seed fixed at 42 throughout.

## 6. Results

### 6.1 Ablation

All figures are on the held-out test set (n = 2,433).

| Configuration | Accuracy | 95% CI | Macro F1 | MCC | Kappa | AUC | Params |
|---|---|---|---|---|---|---|---|
| Base (base-paper reproduction) | 57.95% | [55.99, 59.91] | 0.5995 | 0.5059 | 0.5040 | 0.8932 | 7,324,295 |
| + AFW (N1) | 58.36% | [56.41, 60.32] | 0.5929 | 0.5103 | 0.5078 | 0.8908 | 7,324,458 |
| + EAAA (N2) | 57.62% | [55.66, 59.59] | 0.5944 | 0.5017 | 0.4993 | 0.8903 | 7,324,295 |
| + MSTC (N3) | 59.27% | [57.32, 61.22] | 0.6087 | 0.5210 | 0.5192 | 0.8971 | 7,324,291 |
| + CADL (N4) | 59.10% | [57.15, 61.06] | 0.6038 | 0.5185 | 0.5168 | 0.8888 | 7,324,295 |
| Full (all four) | 55.82% | [53.84, 57.79] | 0.5695 | 0.4825 | 0.4782 | 0.8810 | 7,324,454 |

Change relative to base:

| Novelty | Accuracy delta | Macro F1 delta |
|---|---|---|
| + AFW (N1) | +0.41 pts | -0.0066 |
| + EAAA (N2) | -0.33 pts | -0.0051 |
| + MSTC (N3) | +1.32 pts | +0.0092 |
| + CADL (N4) | +1.15 pts | +0.0044 |
| Full (all four) | -2.14 pts | -0.0300 |

**Statistical caveat.** With n = 2,433 the 95% confidence interval half-width is approximately ±2.0 points. The intervals for all six configurations overlap substantially, so **none of the individual novelty gains is statistically significant at this sample size**. MSTC and CADL are directionally positive and consistent across accuracy, macro F1, MCC and kappa, but this ablation should be read as indicative rather than conclusive. Establishing significance would require repeated runs across seeds — the natural next step for this work.

Two results warrant comment. First, EAAA slightly *reduces* accuracy, suggesting the emotion-conditioned policy constrains augmentation diversity more than it improves class fidelity at this expansion budget. Second, the full configuration is the weakest of the six: combining all four novelties compounds rather than mitigates the overfitting described in Section 6.4.

### 6.2 Per-class performance (base configuration)

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.7220 | 0.5870 | 0.6476 | 385 |
| disgust | 0.6731 | 0.4557 | 0.5435 | 384 |
| fear | 0.4839 | 0.5065 | 0.4949 | 385 |
| happy | 0.5412 | 0.5455 | 0.5433 | 385 |
| neutral | 0.5711 | 0.6675 | 0.6156 | 379 |
| sad | 0.5010 | 0.6494 | 0.5656 | 385 |
| surprise | 0.7953 | 0.7769 | 0.7860 | 130 |

`surprise` achieves the highest F1 despite being the minority class (652 samples). It is absent from CREMA-D, so every surprise sample comes from the acoustically cleaner, less variable corpora. `fear` is weakest, consistent with the base paper's own confusion analysis.

### 6.3 Confusion-pair errors

CADL targets the sad-neutral and angry-fear pairs explicitly.

| Configuration | sad↔neutral | angry↔fear | Total |
|---|---|---|---|
| Base (base-paper reproduction) | 124 | 64 | **188** |
| + AFW (N1) | 114 | 39 | **153** |
| + EAAA (N2) | 128 | 34 | **162** |
| + MSTC (N3) | 135 | 49 | **184** |
| + CADL (N4) | 130 | 42 | **172** |
| Full (all four) | 98 | 61 | **159** |

CADL reduces targeted confusions from 188 to 172 (8.5% fewer), confirming the mechanism works as designed. Unexpectedly, **AFW reduces them further still, to 153** (18.6% fewer) — despite not being designed for that purpose. Per-sample stream reweighting appears to help separate acoustically similar pairs, which is a result worth investigating in future work.

### 6.4 Learning behaviour

The base configuration reaches **98.9% training accuracy against 61.8% validation accuracy** by epoch 14, with validation loss reaching its minimum at **epoch 4** and rising monotonically thereafter. The limiting factor is therefore generalisation, not capacity: 4.85 M of the 7.32 M parameters sit in a single Dense layer over a 9,472-dimensional flatten, trained on 11,700 samples.

### 6.5 AFW interpretability

Mean learned stream weights per emotion (full configuration):

| Emotion | MFCC | ZCR | RMSE |
|---|---|---|---|
| angry | 0.282 | 0.346 | 0.372 |
| disgust | 0.274 | 0.356 | 0.370 |
| fear | 0.282 | 0.357 | 0.361 |
| happy | 0.291 | 0.346 | 0.363 |
| neutral | 0.275 | 0.367 | 0.358 |
| sad | 0.268 | 0.379 | 0.354 |
| surprise | 0.268 | 0.350 | 0.382 |

The gate consistently down-weights MFCC relative to ZCR and RMSE, but the spread across emotions is small (~0.02), indicating the module learns a largely global rebalancing rather than strong per-class specialisation. This is a weaker interpretability result than the proposal anticipated and should be reported as such.

## 7. Reproducibility investigation

The base configuration reproduces the base paper's *pipeline* but not its *result*: 57.95% against 94.91%. Because the implementation is test-verified and the split sizes, class distribution and parameter count all match the paper exactly, we investigated whether the discrepancy is explained by evaluation-protocol defects rather than by implementation differences.

Two controlled experiments were run, each changing exactly one thing against the verified `base` pipeline.

### 7.1 Duplicated dataset mirrors

Scanning the dataset roots rather than the canonical subdirectories yields 16,402 files. Because the duplicates are byte-identical recordings and the split is random over utterances, a clip and its twin land on opposite sides of the boundary with probability 2 x 0.2 x 0.8 = 0.32. Predicted contamination: ~41% of the test set.

**Measured contamination: 40.8%** (1,339 of 3,281 test recordings had a byte-identical twin in train or validation).

Accuracy rose from 57.95% to 72.66% — **+14.71 points**. This is a conservative floor: the duplicated training set already exceeded the augmentation target, so this run trained with *less* augmentation, which depresses accuracy.

### 7.2 Augmenting before splitting

The canonical ordering error in published SER pipelines is to augment every utterance and only then split. Each utterance becomes several near-identical rows; with three rows per utterance a test row has a sibling in training with probability 1 − 0.2² = 0.96. This experiment used the **clean, de-duplicated corpus**, isolating the ordering effect.

**Measured contamination: 96.2%** (7,019 of 7,298 test rows came from an utterance also present in train or validation).

Accuracy rose from 57.95% to 82.23% — **+24.27 points**.

### 7.3 Leakage accounting

| Pipeline | Rows | Contamination | Accuracy | Delta |
|---|---|---|---|---|
| Correct: split then augment | 12,162 | 0% | **57.95%** | — |
| Duplicate mirrors | 16,402 | 40.8% | 72.66% | +14.71 |
| Augment before split | 36,486 | 96.2% | 82.23% | +24.27 |
| *Base paper reported* | *12,162* | *not stated* | *94.91%* | *+36.96* |

Both contamination rates were predicted from first principles before measurement (41% vs 40.8%; 96% vs 96.2%), which supports the mechanism rather than merely the correlation. Neither defect alone reaches 94.91%, but they are independent and compound: treating them as multiplicative on the error rate gives an estimated ~88%, and heavier augmentation than the three rows per utterance used here would raise that further.

**Conclusion.** Published multi-corpus SER accuracies in the mid-90s are reproducible on this data only under evaluation protocols that leak training material into the test set. Under a correct protocol the same architecture achieves 57.95%.

## 8. Discussion

### 8.1 Comparison with the literature

| Study | Corpora | Accuracy |
|---|---|---|
| Dasude et al. (2024) | TESS+RAVDESS+SAVEE+CREMA-D | 50.6% |
| **This work (base, verified protocol)** | same four | **57.95%** |
| **This work (MSTC, best)** | same four | **59.27%** |
| Chourasia et al. (2026), base paper | same four | 94.91% |

Against the closest comparable study — the same four corpora, also reporting a combined-corpus figure — our verified result is 7.4 points higher. CREMA-D constitutes 61% of the fused dataset and is the hardest of the four; published audio-only results on it typically fall in the 60-75% band, which is difficult to reconcile with a 94.91% average across the combination.

### 8.2 Limitations

1. **Statistical power.** Single-seed runs; no novelty gain is significant at n = 2,433. Repeated runs across seeds are required.
2. **Speaker-dependent splitting.** The 72:8:20 split is over utterances, not speakers, so the same speaker appears in train and test. This *inflates* all reported figures. It is retained deliberately for comparability with the base paper; a speaker-independent protocol would be stricter and would lower every number here.
3. **Overfitting is unaddressed.** The 37-point train/validation gap indicates substantial headroom from regularisation alone, which this study did not pursue.
4. **The combined-leakage experiment was not completed**, so the compounded estimate (~88%) remains an extrapolation.

### 8.3 Future work

- Repeated-seed runs with significance testing.
- Regularisation study: the Dense head holds 66% of all parameters.
- Speaker-independent evaluation as a stricter secondary protocol.
- Investigating why AFW reduces confusion-pair errors more than CADL.

## 9. Efficiency study

Section 6.4 identified a 37-point train/validation gap. A follow-up study investigated whether regularisation could close it.

### 9.1 A correction to the overfitting diagnosis

The 37-point gap is measured at **epoch 14**. Training uses `EarlyStopping(restore_best_weights=True)` on validation loss, which restores **epoch 4**, where the gap is only 4.7 points. The *evaluated* model was therefore never badly overfit — early stopping was already performing most of the regularisation. This is why the interventions below produced a modest rather than a dramatic gain, and it corrects the interpretation offered in Section 6.4.

### 9.2 Validation sweep

Six configurations, novelties disabled, scored on **validation only**; the test set was not loaded during selection.

| Configuration | Val accuracy | Params | Δ vs base |
|---|---|---|---|
| **`gap_reg_aug3`** | 60.84% | 2,540,167 | +2.26 |
| `gap_reg` | 60.33% | 2,540,167 | +1.75 |
| `reg` | 59.40% | 7,324,295 | +0.82 |
| `gap` | 58.79% | 2,540,167 | +0.21 |
| `base` | 58.58% | 7,324,295 | +0.00 |
| `gap_reg_aug3_lr` | 58.58% | 2,540,167 | +0.00 |

The winner, `gap_reg_aug3`, combines a GlobalAveragePooling head with dropout 0.35/0.55, L2 1e-4 and 3× augmentation. Notably, the pooling head **on its own** matched base accuracy (58.79% vs 58.58%) using 65% fewer parameters — the `Flatten(9,472) → Dense(512)` head carries 4.85 M parameters and contributes nothing measurable.

### 9.3 Test result

The winner was selected on validation and then evaluated on the test set **once**, alongside the base control.

| Model | Accuracy | 95% CI | Macro F1 | MCC | Kappa | AUC | Params |
|---|---|---|---|---|---|---|---|
| Base | 57.95% | [55.99, 59.91] | 0.5995 | 0.5059 | 0.5040 | 0.8932 | 7,324,295 |
| `gap_reg_aug3` | 60.79% | [58.85, 62.73] | 0.6271 | 0.5381 | 0.5372 | 0.9063 | 2,540,167 |

**+2.84 points with 65% fewer parameters.** Every metric improves together — macro F1 0.5995 → 0.6271, MCC 0.5059 → 0.5381, AUC 0.8932 → 0.9063 — and the confusion-pair errors targeted by CADL fall from 188 to 137 (27.1% fewer) without CADL being enabled.

For a project whose stated goal is a lightweight, edge-deployable model, a smaller *and* more accurate configuration is the more valuable of the two outcomes.

### 9.4 Per-corpus breakdown

| Corpus | n | Base | gap_reg_aug3 |
|---|---|---|---|
| TESS | 553 | 96.75% | **98.55%** |
| RAVDESS | 263 | 47.53% | **55.13%** |
| CREMA-D | 1,513 | 47.12% | **49.90%** |
| SAVEE | 104 | 35.58% | **32.69%** |
| **Combined** | 2,433 | 57.95% | **60.79%** |

This is the most informative table in the report. The same model scores **98.55% on TESS** and **49.90% on CREMA-D**. TESS is two speakers recorded in a studio with deliberately stereotyped delivery, and under the utterance-level split (§8.2) the same speakers appear in training. CREMA-D is 91 crowd-sourced actors with natural delivery and constitutes 62% of the test set.

Two consequences follow. First, the combined figure is essentially a CREMA-D figure. Second — and this bears directly on Section 7 — a single-corpus result on TESS is close to 99% for a model that manages barely 50% on CREMA-D. Published SER accuracies in the mid-90s are therefore explicable not only by the leakage mechanisms of Section 7 but also by **corpus composition**: evaluating on an easy, speaker-dependent corpus and reporting it as a general result.

SAVEE is the weakest at 32.69%, but with only 104 test samples that estimate is noisy, and it is the one corpus where the improved model does not beat base.

## 10. Conclusion

We implemented and ablated four lightweight novelties on a fused four-corpus SER dataset, delivering the component-wise analysis the base paper listed as future work. Three of four novelties improve on the baseline directionally, though none significantly at this sample size; CADL reduces its targeted confusion pairs by 8.5% as designed.

The project's principal contribution is methodological. The base paper's 94.91% could not be reproduced under a verified protocol, and we identify two concrete, independently measured leakage mechanisms — duplicated corpus mirrors and augment-before-split ordering — that inflate accuracy on this data by 14.7 and 24.3 points respectively. Both contamination rates were predicted before measurement and confirmed to within 0.2 points. We therefore report 57.95% as an honest baseline for this corpus combination, exceeding the closest comparable published result by 7.4 points.

The efficiency study adds a third result: 60.79% at 65% of the parameter count, which serves the lightweight-deployment goal better than the accuracy gain alone. The per-corpus breakdown supplies the sharpest single observation in this work — one model, 98.6% on TESS and 49.9% on CREMA-D — and makes clear that any SER accuracy quoted without its corpus composition is close to uninterpretable.

## Appendix A — Artefacts

```
results/ablation/runs/{base,afw,eaaa,mstc,cadl,full}/
    test_metrics.json                 full metric suite
    test_classification_report.csv    per-class precision/recall/F1
    test_confusion_matrix{.csv,.png}  raw and normalised
    training_curves.png               accuracy and loss
    training_log.csv                  per-epoch history
    afw_weights_per_emotion.csv       AFW runs only
results/leak_dup/    duplicate-mirror experiment
results/leak_aug/    augment-before-split experiment
```

Code, tests and notebooks: <https://github.com/Eldorado5002/ser>

## Appendix B — Reproduction

```bash
py -3.10 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest tests/ -v      # 117 tests, ~70 s
```

Kaggle notebooks, in order: `01_features.ipynb` (CPU, feature extraction), `02_train.ipynb` (GPU, six configurations), `03_leakage_test.ipynb`, `04_augment_before_split.ipynb`.

