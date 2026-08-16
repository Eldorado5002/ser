# Adaptive Feature-Weighted 1D-CNN with Emotion-Aware Augmentation for Robust Speech Emotion Recognition

> **Complete project specification for a coding agent.**
> Domain: Speech Processing & Affective Computing · Base paper: Chourasia et al., *Scientific Reports* (2026)
>
> This file contains three parts:
> - **Part A** — the full project proposal (converted from the PDF, all content preserved)
> - **Part B** — the implementation guide: environment, datasets, repo layout, engineering decisions, run commands, and acceptance criteria
> - **Part C** — the complete, ready-to-run source code for every file in the repository

---

## Table of Contents

- [Part A — Project Proposal](#part-a--project-proposal)
  - [1. Problem Statement](#1-problem-statement)
  - [2. Literature Survey](#2-literature-survey)
  - [3. Limitations of Existing Approaches](#3-limitations-of-existing-approaches)
  - [4. Novelty of the Proposed Work](#4-novelty-of-the-proposed-work)
  - [5. Proposed Solution](#5-proposed-solution)
  - [6. Expected Outcome](#6-expected-outcome)
  - [7. Key References](#7-key-references)
- [Part B — Implementation Guide](#part-b--implementation-guide)
- [Part C — Full Source Code](#part-c--full-source-code)

---

# Part A — Project Proposal

## 1. Problem Statement

Speech Emotion Recognition (SER) aims to automatically identify a speaker's emotional state from the acoustic properties of their voice. It is a core enabling technology for human–computer interaction, mental-health assessment, healthcare monitoring, intelligent virtual assistants, call-centre analytics, and human–robot interaction. Despite substantial progress, building an SER system that is simultaneously **accurate, generalizable, and computationally lightweight** remains an open problem.

State-of-the-art deep architectures such as transformers (Wav2Vec2, Audio Spectrogram Transformer) and hybrid CNN–LSTM networks achieve strong accuracy but demand large-scale training data and heavy computational resources, making them unsuitable for real-time or resource-constrained deployment (edge devices, embedded systems, mobile applications). Conversely, lightweight models built on handcrafted acoustic features typically make two simplifying assumptions that limit their performance:

1. **All extracted acoustic features (MFCC, ZCR, RMSE) are treated as equally important** for every utterance, even though the discriminative value of each feature varies from sample to sample and from emotion to emotion.
2. **Identical data augmentation is applied uniformly to all emotion classes**, ignoring that each emotion has distinct natural speech characteristics (pitch range, speaking rate, energy), which can distort the emotional content of augmented samples.

In addition, existing lightweight models rely on **single-scale convolutional kernels** that cannot simultaneously capture short-term acoustic events (energy bursts, micro-pitch variations) and longer-term prosodic patterns (intonation contours, speaking rate), and they employ **standard cross-entropy loss**, which does not explicitly penalize the systematic confusion observed between acoustically similar emotion pairs such as sad–neutral and angry–fear.

**Therefore, the problem addressed by this project is:** to design a Conv1D-based speech emotion recognition framework that retains the computational efficiency of lightweight feature-based models while overcoming the above limitations through *adaptive feature weighting*, *emotion-aware augmentation*, *multi-scale temporal feature extraction*, and *confusion-aware discriminative learning* — evaluated on a combined multi-corpus dataset (RAVDESS, SAVEE, TESS, CREMA-D) across seven emotional categories.

## 2. Literature Survey

### 2.1 Base Paper

**Chourasia, N., Lamba, C. S., & Gupta, A. K. (2026). "A 1D-CNN with advanced data augmentation for robust speech emotion recognition." *Scientific Reports*.**

The base paper proposes a lightweight one-dimensional convolutional neural network (1D-CNN) for SER. Three complementary handcrafted acoustic features are extracted frame-wise from each 3-second utterance — 40 Mel-Frequency Cepstral Coefficients (MFCC) capturing the spectral envelope, Zero-Crossing Rate (ZCR) capturing temporal signal transitions, and Root Mean Square Energy (RMSE) capturing loudness dynamics — and concatenated into a unified sequential vector of shape (2376, 1). The network consists of five stacked Conv1D blocks (512→512→256→256→128 filters), each followed by batch normalization and max-pooling, then dense layers and a 7-way Softmax classifier (≈7.19 M parameters). Four benchmark corpora — RAVDESS, SAVEE, TESS, and CREMA-D — are fused into a combined dataset of 12,162 samples (72:8:20 train/validation/test split), and four augmentation techniques (Gaussian noise injection, time shifting, pitch modification, time stretching) are applied only to the training subset. The model achieves **94.91% test accuracy** and a **Macro F1-score of 0.94**, with strong statistical reliability (MCC 0.9294, Cohen's Kappa 0.9292, AUC 0.9963), demonstrating that a well-designed lightweight CNN with discriminative features can rival far heavier architectures.

### 2.2 Related Work

Research in SER has evolved along three major directions: (i) handcrafted features with classical machine learning, (ii) end-to-end deep learning architectures, and (iii) augmentation-based strategies for data scarcity and class imbalance. Representative studies are summarized in Table 1.

| Study | Method / Features | Dataset(s) | Reported Result | Limitation |
|---|---|---|---|---|
| Krishna et al. (2022) [18] | MFCC, MEL, Chroma + SVM / MLP | SER speech datasets | Improved accuracy with hybrid features | Requires manual feature engineering; limited representation power |
| Badshah et al. (2017) [19] | Spectrograms + CNN with transfer learning (AlexNet) | Berlin EMO-DB | High accuracy for most emotions | Poor performance on some emotions (fear); small dataset |
| Meng et al. (2019) [20] | ADRNN: dilated CNN + residual + BiLSTM + attention | IEMOCAP, EMO-DB | 63.84% accuracy | Complex architecture; high computational cost |
| Zhao et al. (2019) [23] | Hybrid 1D/2D CNN–LSTM | RAVDESS | 90.2% accuracy, F1 0.90 | Increased model complexity; single-dataset evaluation |
| Pepino et al. (2021) [24] | Wav2Vec2 transformer embeddings | Multi-dataset | 92.0% accuracy, strong generalization | Needs large pretrained models; high computational cost |
| Kilimci et al. (2025) [26] | End-to-end CNN on raw audio | SER benchmarks | Up to 99.46% accuracy | Overfitting risk; dependent on dataset characteristics |
| Ahmed et al. (2023) [28] | Ensemble 1D-CNN–LSTM–GRU + augmentation | Multi-dataset (multilingual) | 93.5% accuracy, F1 0.93 | High training complexity and resource demands |
| Saloumi et al. (2023) [29] | MFCC + 1D-CNN with augmentation | RAVDESS | 83% accuracy | Small dataset; sensitive to data diversity; MFCC-only |
| Baek & Lee (2023) [4] | DCGAN-based synthetic augmentation + CNN | SER datasets | Improved robustness | GAN training instability; synthetic data reliability concerns |
| Dasude et al. (2024) [43] | CNN, LSTM, ConvLSTM comparison | TESS+RAVDESS+SAVEE+CREMA-D | 50.6% on combined corpus | Moderate accuracy under dataset heterogeneity |
| Akinpelu et al. (2024) [66] | Vision Transformer (ViT) | RAVDESS | 93.8% accuracy | Large data requirements; heavy architecture |
| Bhanbhro et al. (2025) [67] | Attention-enhanced CNN–LSTM | RAVDESS | 94.2% accuracy | Higher complexity; single-corpus evaluation |
| Tang et al. (2025) [68] | CNN-Transformer + multidimensional attention | MELD, RAVDESS | 95.2% accuracy (highest) | Significantly larger model; heavy computational resources |
| **Base paper — Chourasia et al. (2026)** | MFCC+ZCR+RMSE + 5-block 1D-CNN + uniform augmentation | RAVDESS+SAVEE+TESS+CREMA-D | 94.91% accuracy, Macro F1 0.94 | Equal feature treatment; uniform augmentation; single-scale kernels; plain cross-entropy loss |

*Table 1: Comparative summary of representative SER studies.*

Collectively, the literature shows a clear trade-off: transformer- and attention-based models (Tang et al., Bhanbhro et al., Akinpelu et al.) push accuracy upward but at substantial computational expense, while lightweight feature-based models (Saloumi et al., base paper) remain deployable but leave measurable performance on the table due to static feature fusion and uniform augmentation. **This project targets precisely that gap.**

## 3. Limitations of Existing Approaches

- **L1 — Equal (static) feature treatment.** Existing lightweight SER pipelines, including the base paper, simply concatenate MFCC, ZCR, and RMSE, implicitly assuming each feature contributes equally to every prediction. In reality, some utterances are tone-distinctive (MFCC-dominant) while others are energy-distinctive (RMSE-dominant); static concatenation prevents the network from exploiting this per-sample variability.

- **L2 — Emotion-agnostic, uniform augmentation.** The same four augmentation techniques are applied identically to all seven emotion classes. Because each emotion has characteristic acoustics (happy: wide pitch variation; sad: slow rate and low energy; angry: high energy; fear: erratic fluctuations), uniform augmentation can push augmented samples away from the natural acoustic manifold of their class, injecting label noise instead of useful diversity.

- **L3 — Single-scale temporal receptive fields.** Each convolutional layer in existing 1D-CNN pipelines uses a single kernel size, so any one layer observes emotional cues at only one temporal resolution. Short-lived events (energy bursts, micro-pitch variations) and slower prosodic structures (intonation contours, stress patterns) are therefore never captured in parallel at the same depth.

- **L4 — Confusion-blind training objective.** Models are trained with standard categorical cross-entropy, which treats all misclassifications as equally costly. Yet confusion matrices consistently show that errors concentrate in acoustically similar pairs — sad↔neutral (both low-energy, slow) and angry↔fear (both high-arousal, intense). Nothing in the objective explicitly encourages the model to separate these hard pairs.

- **L5 — Heavy alternatives are impractical for deployment.** Transformer, ViT, and hybrid recurrent architectures that could address the above issues require large pretrained models, large-scale datasets, and high computational resources, contradicting the goal of real-time SER on edge and embedded devices.

- **L6 — Limited evaluation practice.** Several existing studies evaluate on a single corpus (often RAVDESS), overstating generalization; ablation analyses of individual features and augmentations are usually missing, so the contribution of each component remains unquantified.

## 4. Novelty of the Proposed Work

This project retains the efficient Conv1D backbone of the base paper and introduces **four lightweight, mutually complementary novelties**. Each novelty directly answers one of the limitations identified in Section 3, and each can be enabled or disabled independently, giving the project a natural ablation-study structure that the base paper itself lacked.

### Novelty 1 — Adaptive Feature Weighting (AFW) [addresses L1]

Instead of concatenating MFCC, RMSE, and ZCR with fixed equal importance, AFW inserts a small attention-style gating module between feature extraction and the Conv1D backbone. The module takes the three feature streams, computes a per-sample importance score for each (compact dense layer followed by a softmax), and scales every stream by its learned weight before fusion. A tone-distinctive utterance may thus receive weights of approximately 60% MFCC / 25% RMSE / 15% ZCR, while an energy-distinctive utterance shifts emphasis toward RMSE. The weights are trained jointly with the network, add only a few thousand parameters, and preserve the lightweight design goal.

### Novelty 2 — Emotion-Aware Adaptive Augmentation (EAAA) [addresses L2]

EAAA replaces uniform augmentation with an emotion-conditioned augmentation policy: the technique applied to each training sample is selected to mirror the natural acoustic variability of its emotion class, so augmented data stays faithful to how that emotion actually sounds. The mapping, grounded in the speech-emotion literature and in the base paper's own spectrogram analysis, is given in Table 2.

| Emotion | Natural Acoustic Characteristic | Assigned Augmentation |
|---|---|---|
| Happy | Wide pitch variation; expressive, rhythmic delivery | Pitch shifting (±2 semitones) |
| Sad | Slow speaking rate; low, smooth energy | Time stretching (0.8–1.2×) |
| Angry | High energy, vocal strain, intense peaks | Noise injection (0.001–0.005) |
| Fear | Erratic fluctuations; unstable low-to-mid energy | Time shifting (±0.2 s) + light noise |
| Surprise | Sudden bursts of spectral energy; brief utterances | Time shifting (±0.2 s) |
| Disgust | Controlled, tense, low-frequency dominance | Mild pitch shift + mild stretch |
| Neutral | Steady, evenly distributed spectral energy | Light noise injection only |

*Table 2: Emotion-to-augmentation mapping used by EAAA (all seven classes).*

### Novelty 3 — Multi-Scale Temporal Convolution (MSTC) [addresses L3]

The first convolutional block is replaced by three parallel Conv1D branches with kernel sizes 3, 5, and 7 operating on the same input, whose outputs are concatenated channel-wise before entering the remaining backbone. The small kernel captures rapid, short-term acoustic events (energy bursts, micro-pitch variation), the medium kernel captures phoneme-level spectral transitions, and the large kernel captures slower prosodic structure (intonation contours, speaking-rate patterns) — all at the same network depth. Because the total filter budget of the block is split across the three branches rather than tripled, the parameter count remains close to the original single-scale block, preserving computational efficiency.

### Novelty 4 — Confusion-Aware Discriminative Loss (CADL) [addresses L4]

Standard categorical cross-entropy is replaced by a composite objective: cross-entropy combined with a focal-style modulation and an additional pairwise penalty term applied to the empirically confusable emotion pairs (sad↔neutral and angry↔fear). The focal component down-weights easy, already-correct samples so training concentrates on hard examples, while the pairwise penalty adds an extra margin cost whenever a sample is misclassified into its known confusion partner. This directly targets the dominant error mode reported in the base paper's confusion matrix **without adding any inference-time cost** — the network architecture and prediction pipeline are unchanged; only training is affected.

| Novelty | Limitation Addressed | Where It Acts | Cost |
|---|---|---|---|
| N1: Adaptive Feature Weighting (AFW) | L1 — equal feature treatment | Feature-fusion level | Few thousand extra parameters (<0.1%) |
| N2: Emotion-Aware Adaptive Augmentation (EAAA) | L2 — uniform augmentation | Data level (training only) | Zero model overhead; rule-based selection |
| N3: Multi-Scale Temporal Convolution (MSTC) | L3 — single-scale kernels | Architecture level (first block) | ≈ parameter-neutral (filter budget split) |
| N4: Confusion-Aware Discriminative Loss (CADL) | L4 — confusion-blind objective | Training-objective level | Zero inference-time overhead |

*Table 3: Summary of the four proposed novelties.*

## 5. Proposed Solution

The proposed system is an **Adaptive Feature-Weighted, Multi-Scale 1D-CNN with Emotion-Aware Augmentation and Confusion-Aware Training**. It follows the base paper's efficient pipeline end-to-end, upgrading four specific stages.

### 5.1 Dataset and Preprocessing

Four public benchmark corpora are fused into a single multi-speaker dataset: **RAVDESS** (1,440 speech samples, 24 actors), **TESS** (2,800 samples, 2 female speakers), **SAVEE** (480 samples, male speakers), and **CREMA-D** (7,442 samples, 91 actors) — **12,162 samples in total**, mapped to seven emotion classes (disgust, happy, sad, neutral, fear, angry, surprise). All audio is resampled to 22,050 Hz, amplitude-normalized to [−1, 1], and zero-padded or truncated to a fixed 3-second duration. The data is split **72:8:20** into training (8,756), validation (973), and test (2,433) subsets **before augmentation**, guaranteeing that validation and test sets remain untouched and preventing data leakage.

### 5.2 Emotion-Aware Augmentation Stage (EAAA)

Augmentation is applied **only to the training subset**. For each training sample, the augmentation technique is selected according to the emotion-to-augmentation policy of Table 2, expanding the effective training set (to approximately 11,700 samples, matching the base paper's expansion budget for a fair comparison) while keeping every augmented sample acoustically consistent with its emotion class.

### 5.3 Feature Extraction and Adaptive Weighting (AFW)

For every 3-second sample, frame-level features are computed with a 25 ms window and 10 ms hop: 40 MFCC coefficients, ZCR, and RMSE. Before fusion, the AFW module computes per-sample softmax importance weights over the three streams and rescales each stream accordingly; the weighted streams are then concatenated into the sequential input tensor of shape (2376, 1) expected by the backbone. The learned weights are also interpretable — they can be visualized per emotion to show which acoustic cue drives each class.

### 5.4 Multi-Scale Conv1D Backbone (MSTC)

The backbone follows the base architecture — five convolutional stages with batch normalization and max-pooling (pool size 2), filters decreasing 512→512→256→256→128 — with the first stage replaced by the multi-scale block: three parallel Conv1D branches (kernel sizes 3, 5, 7) whose outputs are concatenated channel-wise. The final feature maps are flattened and passed through a Dense(512) layer with ReLU and batch normalization, ending in a 7-neuron Softmax output. Total parameters remain close to the base model's ≈7.19 M, keeping the model deployable on resource-constrained hardware.

### 5.5 Confusion-Aware Training (CADL)

The network is trained with the Adam optimizer (learning rate 0.001, batch size 32, 50 epochs) using the composite CADL objective: categorical cross-entropy with focal modulation plus the pairwise margin penalty on the sad↔neutral and angry↔fear pairs. Early stopping on validation loss guards against overfitting.

### 5.6 Evaluation Protocol

Performance is measured on the held-out test set using accuracy, per-class and macro precision/recall/F1, specificity, G-Mean, MCC, Cohen's Kappa, AUC, confusion matrices, and a 95% confidence interval for accuracy — the same metric suite as the base paper, enabling direct comparison. In addition, a **component-wise ablation study** is conducted (base model, +AFW, +EAAA, +MSTC, +CADL, and all combined) to quantify the individual contribution of each novelty — an analysis explicitly missing from the base paper.

### 5.7 Workflow Summary

| Step | Stage | Description |
|---|---|---|
| 1 | Data fusion | Merge RAVDESS + SAVEE + TESS + CREMA-D; map to 7 emotion classes (12,162 samples) |
| 2 | Preprocessing | Resample 22,050 Hz → normalize [−1,1] → pad/truncate to fixed duration |
| 3 | Split | 72:8:20 train / validation / test split (before augmentation) |
| 4 | EAAA (Novelty 2) | Emotion-conditioned augmentation applied to training data only |
| 5 | Feature extraction | MFCC + ZCR + RMSE per frame |
| 6 | AFW (Novelty 1) | Per-sample learned weighting of the three feature streams before fusion |
| 7 | MSTC backbone (Novelty 3) | Multi-scale first block (kernels 3/5/7) + 4 further Conv1D stages |
| 8 | CADL training (Novelty 4) | Focal cross-entropy + pairwise confusion penalty; Adam, lr 0.001 |
| 9 | Evaluation | Full metric suite + confusion matrix + ablation study |

*Table 4: End-to-end workflow of the proposed framework.*

## 6. Expected Outcome

- **Improved recognition accuracy over the base paper.** By resolving static feature fusion, uniform augmentation, single-scale receptive fields, and the confusion-blind objective simultaneously, the framework is expected to exceed the base paper's 94.91% test accuracy and 0.94 Macro F1 on the identical combined-corpus test protocol, with a target in the range of **95.5–96.5%**.
- **Reduced confusion between hard emotion pairs.** CADL is expected to measurably lower the sad↔neutral and angry↔fear misclassification counts in the confusion matrix — the dominant error mode of the base model — improving per-class F1 for sad and neutral (currently the weakest at 0.93).
- **More realistic, class-faithful augmented data.** EAAA is expected to raise the quality of augmentation, particularly benefiting minority classes such as surprise (absent from CREMA-D), improving robustness without injecting label noise.
- **Interpretability of acoustic cues.** The AFW module yields per-sample and per-class feature-importance weights, providing insight into which acoustic dimension (spectral, temporal, or energy) drives each emotion — a form of built-in explainability absent from the base model.
- **Preserved computational efficiency.** All four novelties are lightweight: total parameters remain approximately at the base model's 7.19 M, and inference cost is unchanged (CADL and EAAA affect training only), keeping the system suitable for real-time and edge deployment.
- **A complete ablation analysis.** The project will deliver the component-wise ablation (base, +AFW, +EAAA, +MSTC, +CADL, all combined) that the base paper listed as missing future work, quantifying each novelty's individual contribution.
- **Deliverables.** A trained SER model, full source code, evaluation reports (metric tables, training curves, confusion matrices, ablation tables), and a project report suitable for academic submission.

## 7. Key References

1. Chourasia, N., Lamba, C. S., & Gupta, A. K. (2026). A 1D-CNN with advanced data augmentation for robust speech emotion recognition. *Scientific Reports*. https://doi.org/10.1038/s41598-026-56241-x **[Base Paper]**
2. Zhao, J., Mao, X., & Chen, L. (2019). Speech emotion recognition using deep 1D and 2D CNN-LSTM networks. *Biomedical Signal Processing and Control*, 47, 312–323.
3. Pepino, L., Riera, P., & Ferrer, L. (2021). Emotion recognition from speech using wav2vec 2.0 embeddings. arXiv:2104.03502.
4. Ahmed, M. R., Islam, S., Islam, A. M., & Shatabda, S. (2023). An ensemble 1D-CNN-LSTM-GRU model with data augmentation for speech emotion recognition. *Expert Systems with Applications*, 218, 119633.
5. Saloumi, M., et al. (2023). Speech emotion recognition using one-dimensional convolutional neural networks. *Proc. 46th TSP*, 212–216. IEEE.
6. Baek, J.-Y., & Lee, S.-P. (2023). Enhanced speech emotion recognition using DCGAN-based data augmentation. *Electronics*, 12(18), 3966.
7. Badshah, A. M., Ahmad, J., Rahim, N., & Baik, S. W. (2017). Speech emotion recognition from spectrograms with deep convolutional neural network. *Proc. PlatCon*, 1–5. IEEE.
8. Krishna, K. V., Sainath, N., & Posonia, A. M. (2022). Speech emotion recognition using machine learning. *Proc. 6th ICCMC*, 1014–1018. IEEE.
9. Akinpelu, O., Ezin, E. C., & Kpalma, K. (2024). An enhanced speech emotion recognition using vision transformer. *Scientific Reports*, 14, 63776.
10. Bhanbhro, J., Shaikh, A., Memon, S., & Rajput, D. (2025). Speech emotion recognition: Comparative analysis of attention-enhanced CNN-LSTM architectures. *Signals*, 6(2), 22.
11. Tang, X., Zhang, Y., Liu, H., & Wang, J. (2025). Speech emotion recognition via CNN-transformer and multidimensional attention mechanism. *Computer Speech & Language*, 91, 101746.
12. Livingstone, S. R., & Russo, F. A. (2018). The Ryerson audio-visual database of emotional speech and song (RAVDESS). *PLOS ONE*, 13(5), e0196391.
13. Pichora-Fuller, M. K., & Dupuis, K. (2020). Toronto emotional speech set (TESS). Scholars Portal Dataverse.
14. Jackson, P., & Haq, S. (2014). Surrey audio-visual expressed emotion (SAVEE) database. University of Surrey.
15. Cao, H., et al. (2014). CREMA-D: Crowd-sourced emotional multimodal actors dataset. *IEEE Transactions on Affective Computing*, 5(4), 377–390.
16. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *Proc. ICCV*, 2980–2988.

---

# Part B — Implementation Guide

Everything a coding agent needs to build, run, and validate this project.

## B.1 Tech Stack

- **Python** ≥ 3.10
- **TensorFlow / Keras** ≥ 2.15 (Conv1D backbone, custom layer, custom loss)
- **librosa** ≥ 0.10 (audio loading, MFCC/ZCR/RMSE, pitch shift, time stretch)
- **scikit-learn** (stratified splits, StandardScaler, full metric suite)
- **numpy, pandas, matplotlib, tqdm, soundfile, joblib**

Install with: `pip install -r requirements.txt`. A GPU is strongly recommended for the 50-epoch runs (each full run is roughly 30–60 min on a modern GPU, several hours on CPU); the pipeline works on CPU too.

## B.2 Datasets — Download and Directory Layout

Download the four corpora (all free; some require registration or a Kaggle account) and place them under `data/`:

| Corpus | Samples used | Source |
|---|---|---|
| RAVDESS (speech, audio-only) | 1,440 | https://zenodo.org/record/1188976 (file `Audio_Speech_Actors_01-24.zip`) — also on Kaggle |
| TESS | 2,800 | https://tspace.library.utoronto.ca/handle/1807/24487 — also on Kaggle ("Toronto emotional speech set") |
| SAVEE | 480 | http://kahlan.eps.surrey.ac.uk/savee/ — also on Kaggle ("Surrey Audio-Visual Expressed Emotion") |
| CREMA-D | 7,442 | https://github.com/CheyneyComputerScience/CREMA-D (`AudioWAV/`) — also on Kaggle |

**Expected layout** (any nesting works — the scanner is recursive; only the top-level folder names must match `config.py`):

```tree
---
ser-project/
├── data/
│   ├── RAVDESS/
│   │   └── Actor_01 ... Actor_24/   (03-01-06-01-02-01-12.wav ...)
│   ├── TESS/
│   │   └── OAF_angry ... YAF_sad/   (OAF_back_angry.wav ...)
│   ├── SAVEE/
│   │   └── (DC_a01.wav ... or speaker folders)
│   └── CREMA-D/
│       └── AudioWAV/                (1091_DFA_ANG_XX.wav ...)
---
```

**Label parsing rules implemented in `data_loader.py`:**

- **RAVDESS**: 3rd hyphen-field of the filename → `01` neutral, `02` calm (**mapped to neutral** so all 1,440 samples are kept and the 12,162 total matches the base paper), `03` happy, `04` sad, `05` angry, `06` fear, `07` disgust, `08` surprise.
- **TESS**: last underscore token of the filename → emotion name; `ps` / `pleasant_surprise` → surprise.
- **SAVEE**: leading letters of the utterance code → `a` angry, `d` disgust, `f` fear, `h` happy, `n` neutral, `sa` sad, `su` surprise (two-letter codes checked first).
- **CREMA-D**: 3rd underscore token → `ANG, DIS, FEA, HAP, NEU, SAD` (CREMA-D has no surprise class).

Verify the scan with `python data_loader.py` — it should report ≈12,162 samples total.

## B.3 Repository Layout

```tree
---
ser-project/
├── PROJECT_SPEC.md      (this file)
├── README.md            (quickstart)
├── requirements.txt
├── config.py            (all hyperparameters / paths / novelty settings)
├── data_loader.py       (corpus scan + label parsing + 72:8:20 split)
├── augmentation.py      (EAAA policy + uniform baseline)     [Novelty 2]
├── features.py          (MFCC/ZCR/RMSE streams + caching)
├── model.py             (AFW gate + MSTC block + backbone)   [Novelties 1, 3]
├── losses.py            (CADL composite loss)                [Novelty 4]
├── train.py             (end-to-end pipeline, CLI flags per novelty)
├── evaluate.py          (full metric suite + AFW interpretability)
├── ablation.py          (6-configuration ablation runner)
├── data/                (datasets - user supplied, see B.2)
├── features_cache/      (auto-generated .npz feature caches)
└── runs/                (auto-generated experiment artefacts)
---
```

## B.4 Engineering Decisions the Agent Must Know

1. **Input-shape reconciliation (important).** The proposal prose says "3 s, 25 ms window, 10 ms hop, 40 MFCC" *and* an input of shape (2376, 1). These are mutually inconsistent — the (2376, 1) shape of the base paper is exactly reproduced by: 2.5 s of audio (0.6 s offset) at 22,050 Hz, `frame_length=2048`, `hop_length=512` → 108 frames; 20 MFCCs flattened (20×108 = 2160) + ZCR (108) + RMSE (108) = **2376**. The code defaults to these shape-reproducing settings for a fair comparison with the base paper, but every value is a `config.py` constant and the model input length is computed dynamically, so switching to the literal 3 s / 40-MFCC configuration requires only editing `config.py` (and clearing `features_cache/`).
2. **Streams stay separate until inside the model.** AFW (Novelty 1) must weight MFCC/ZCR/RMSE individually, so `features.py` returns three arrays and the Keras model takes three inputs; fusion (weighted or static concat) happens as the first model layer. Static concatenation of unweighted streams is exactly the base-paper behaviour and is used when `--no-afw`.
3. **AFW weight scaling.** The softmax weights are multiplied by 3 before scaling the streams so the equal-importance solution (⅓, ⅓, ⅓) leaves magnitudes unchanged — this stabilises early training and makes the base/AFW comparison clean.
4. **MSTC filter budget.** The first stage's 512 filters are split 172/170/170 across the k=3/5/7 branches (remainder absorbed by the first branch), keeping parameters ≈ equal to the single-scale stage as required by Table 3.
5. **CADL formulation.** `loss = focal_CE(γ=2) + λ·Σ_pairs [ y_a·(−log(1−p_b)) + y_b·(−log(1−p_a)) ]` with λ=0.5 and pairs sad↔neutral, angry↔fear (indices resolved from `config.EMOTIONS`, which is alphabetical: angry, disgust, fear, happy, neutral, sad, surprise). With focal and pairwise both disabled it reduces exactly to plain cross-entropy.
6. **Split before augmentation.** The 72:8:20 stratified split happens on the raw metadata; only the training partition is ever augmented (no leakage). The augmented training set is grown to `TARGET_TRAIN_SIZE = 11,700` total samples to match the proposal's expansion budget; set it to `None` for a simple 2× expansion.
7. **Reproducible augmentation.** Every augmented copy carries its own RNG seed, so a given experiment tag always produces identical features (and the on-disk cache stays valid).
8. **Per-stream standardisation.** A separate `StandardScaler` per stream, fit on the (augmented) training set only; scalers are saved to `runs/<tag>/scalers.joblib` for inference reuse.
9. **Feature caching.** Extraction over ~15k clips takes a while, so feature matrices are cached in `features_cache/*.npz`, keyed by an MD5 fingerprint of the item list + feature settings. Delete the folder after changing any audio/feature constant.
10. **Time stretching changes waveform length** — every augmented waveform is re-fixed to `N_SAMPLES` before feature extraction.

## B.5 How to Run

```bash
pip install -r requirements.txt
python data_loader.py                                   # sanity-check dataset scan
python train.py --tag full                              # proposed model (all 4 novelties)
python train.py --tag base --no-afw --no-eaaa --no-mstc --no-cadl   # base-paper reproduction
python train.py --tag afw_only --no-eaaa --no-mstc --no-cadl        # any single novelty
python ablation.py                                      # full 6-way ablation study
python ablation.py --epochs 10                          # quick sanity sweep
python evaluate.py runs/full                            # re-evaluate a saved run
```

Every run writes to `runs/<tag>/`: `best_model.keras`, `scalers.joblib`, `training_log.csv`, `training_curves.png`, `test_metrics.json`, `test_classification_report.csv`, `test_confusion_matrix{.csv,.png,_norm.png}`, and (when AFW is on) `afw_weights_per_emotion.csv`. The ablation summary is `runs/ablation_results.csv`.

## B.6 Acceptance Criteria / Task Checklist for the Coding Agent

- [ ] `python data_loader.py` reports ≈12,162 samples (1,440 RAVDESS + 2,800 TESS + 480 SAVEE + 7,442 CREMA-D) across exactly 7 classes.
- [ ] Split sizes ≈ 8,756 / 973 / 2,433 (72:8:20), stratified, computed before augmentation.
- [ ] With default config, the fused input length is exactly 2376 (asserted by `config.INPUT_LEN`).
- [ ] EAAA applies the Table 2 policy per class; `--no-eaaa` falls back to the uniform 4-technique pool.
- [ ] Base configuration (`--no-afw --no-eaaa --no-mstc --no-cadl`) reproduces the base-paper pipeline; target ≈94–95% test accuracy.
- [ ] Full configuration targets **95.5–96.5%** test accuracy and Macro F1 > 0.94.
- [ ] `runs/<tag>/test_metrics.json` contains accuracy + 95% CI, macro P/R/F1, per-class specificity, G-mean, MCC, Cohen's Kappa, macro OvR AUC, and the sad↔neutral / angry↔fear error counts.
- [ ] CADL runs show reduced confusion-pair error counts versus base.
- [ ] `runs/ablation_results.csv` contains all six configurations with parameter counts (all ≈7–8 M; AFW adds <0.1%).
- [ ] AFW runs produce `afw_weights_per_emotion.csv` (interpretability deliverable).

---

# Part C — Full Source Code

Every file below is complete and ready to run — create the files exactly as shown (or use the accompanying `ser-project.zip`, which contains them already).

## C.1 `requirements.txt`

Python dependencies.

```text
tensorflow>=2.15
librosa>=0.10
numpy>=1.26
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.8
soundfile>=0.12
tqdm>=4.66
joblib>=1.3
```

## C.2 `config.py`

All hyperparameters, paths, emotion classes and novelty settings.

```python
"""Central configuration for the SER project.

Every tunable used anywhere in the pipeline lives here so the coding agent
(and ablation runner) can toggle behaviour from one place.
"""
import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = os.environ.get("SER_DATA_DIR", "data")
RAVDESS_DIR = os.path.join(DATA_DIR, "RAVDESS")
TESS_DIR = os.path.join(DATA_DIR, "TESS")
SAVEE_DIR = os.path.join(DATA_DIR, "SAVEE")
CREMAD_DIR = os.path.join(DATA_DIR, "CREMA-D")

CACHE_DIR = "features_cache"   # cached feature matrices (.npz)
RUNS_DIR = "runs"              # one sub-folder per experiment/tag

# --------------------------------------------------------------------------
# Audio / preprocessing
# --------------------------------------------------------------------------
# IMPORTANT IMPLEMENTATION NOTE
# The base paper reports a fused input vector of shape (2376, 1). That exact
# length is reproduced by the settings below:
#   2.5 s of audio (offset 0.6 s) at 22,050 Hz -> 55,125 samples
#   frame_length = 2048, hop_length = 512      -> 108 frames (center=True)
#   20 MFCCs flattened (20 x 108 = 2160) + ZCR (108) + RMSE (108) = 2376
# The proposal prose also mentions "3 s / 25 ms window / 10 ms hop / 40 MFCC",
# which produces a much longer vector. This codebase computes the input length
# DYNAMICALLY from the values below, so either configuration trains correctly;
# the defaults reproduce the base paper's (2376, 1) input for fair comparison.
SAMPLE_RATE = 22050
DURATION = 2.5          # seconds kept from each clip
OFFSET = 0.6            # seconds skipped at the start (leading silence)
FRAME_LENGTH = 2048     # samples per analysis frame
HOP_LENGTH = 512        # samples between frames
N_MFCC = 20             # number of MFCC coefficients

N_SAMPLES = int(SAMPLE_RATE * DURATION)          # 55125
N_FRAMES = 1 + N_SAMPLES // HOP_LENGTH           # 108   (librosa center=True)
MFCC_LEN = N_MFCC * N_FRAMES                     # 2160
ZCR_LEN = N_FRAMES                               # 108
RMSE_LEN = N_FRAMES                              # 108
INPUT_LEN = MFCC_LEN + ZCR_LEN + RMSE_LEN        # 2376 with the defaults

# --------------------------------------------------------------------------
# Emotion classes (alphabetical order = class index order everywhere)
# --------------------------------------------------------------------------
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}
ID_TO_EMOTION = {i: e for e, i in EMOTION_TO_ID.items()}

# Empirically confusable pairs targeted by the CADL loss (Novelty 4)
CONFUSION_PAIRS = [("sad", "neutral"), ("angry", "fear")]

# --------------------------------------------------------------------------
# Dataset split  (72 : 8 : 20 as in the base paper)
# --------------------------------------------------------------------------
TEST_FRACTION = 0.20            # 20% held-out test
VAL_FRACTION_OF_TRAINVAL = 0.10  # 10% of the remaining 80%  -> 8% overall
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Augmentation (EAAA - Novelty 2)
# --------------------------------------------------------------------------
# Total size of the augmented training set (originals + augmented copies).
# The proposal matches the base paper's expansion budget of ~11,700 training
# samples. Set to None to instead create exactly one augmented copy per
# training sample (2x expansion).
TARGET_TRAIN_SIZE = 11700

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
CONV_FILTERS = [512, 512, 256, 256, 128]   # five stages, base paper backbone
BASE_KERNEL_SIZE = 5                       # kernel used by single-scale blocks
MSTC_KERNELS = (3, 5, 7)                   # parallel kernels of Novelty 3
DENSE_UNITS = 512
DROPOUT_CONV = 0.20
DROPOUT_DENSE = 0.30
AFW_HIDDEN_UNITS = 16                      # hidden units of the AFW gate

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# --------------------------------------------------------------------------
# CADL loss (Novelty 4)
# --------------------------------------------------------------------------
FOCAL_GAMMA = 2.0     # focal modulation strength
PAIR_LAMBDA = 0.5     # weight of the pairwise confusion penalty
```

## C.3 `data_loader.py`

Corpus scanning, per-dataset label parsing, and the stratified 72:8:20 split.

```python
"""Dataset discovery and label parsing for RAVDESS, TESS, SAVEE and CREMA-D.

Builds a single pandas DataFrame with columns:
    path     - absolute path to the .wav file
    emotion  - one of config.EMOTIONS
    corpus   - which dataset the file came from

Filename conventions handled
----------------------------
RAVDESS : 03-01-06-01-02-01-12.wav ; 3rd field is the emotion code
          01 neutral, 02 calm (mapped to neutral), 03 happy, 04 sad,
          05 angry, 06 fearful, 07 disgust, 08 surprised
TESS    : OAF_back_angry.wav / YAF_dog_ps.wav ; last "_" token is the emotion
          ("ps" / "pleasant_surprise" -> surprise)
SAVEE   : DC_a01.wav or a01.wav inside speaker folders ;
          leading letters encode the emotion: a, d, f, h, n, sa, su
CREMA-D : 1091_DFA_ANG_XX.wav ; 3rd "_" token in
          {ANG, DIS, FEA, HAP, NEU, SAD}  (no surprise in CREMA-D)
"""
from __future__ import annotations

import os
import glob
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Per-corpus parsers: return an emotion string from config.EMOTIONS, or None
# ---------------------------------------------------------------------------
_RAVDESS_CODE = {
    "01": "neutral", "02": "neutral",  # calm merged into neutral (7-class setup)
    "03": "happy", "04": "sad", "05": "angry",
    "06": "fear", "07": "disgust", "08": "surprise",
}

_TESS_MAP = {
    "angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
    "neutral": "neutral", "sad": "sad",
    "ps": "surprise", "pleasant_surprise": "surprise", "surprise": "surprise",
}

_SAVEE_MAP = {
    "a": "angry", "d": "disgust", "f": "fear", "h": "happy",
    "n": "neutral", "sa": "sad", "su": "surprise",
}

_CREMAD_MAP = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}


def _parse_ravdess(fname: str) -> str | None:
    parts = os.path.basename(fname).replace(".wav", "").split("-")
    if len(parts) < 3:
        return None
    return _RAVDESS_CODE.get(parts[2])


def _parse_tess(fname: str) -> str | None:
    stem = os.path.basename(fname).replace(".wav", "").lower()
    token = stem.split("_")[-1]
    return _TESS_MAP.get(token)


def _parse_savee(fname: str) -> str | None:
    stem = os.path.basename(fname).replace(".wav", "")
    # Kaggle version prefixes the speaker: "DC_a01" -> take part after "_"
    if "_" in stem:
        stem = stem.split("_")[-1]
    letters = "".join(ch for ch in stem if ch.isalpha()).lower()
    # two-letter codes first so "sa"/"su" are not read as a single letter
    if letters[:2] in _SAVEE_MAP:
        return _SAVEE_MAP[letters[:2]]
    if letters[:1] in _SAVEE_MAP:
        return _SAVEE_MAP[letters[:1]]
    return None


def _parse_cremad(fname: str) -> str | None:
    parts = os.path.basename(fname).replace(".wav", "").split("_")
    if len(parts) < 3:
        return None
    return _CREMAD_MAP.get(parts[2].upper())


_CORPORA = [
    ("RAVDESS", config.RAVDESS_DIR, _parse_ravdess),
    ("TESS", config.TESS_DIR, _parse_tess),
    ("SAVEE", config.SAVEE_DIR, _parse_savee),
    ("CREMA-D", config.CREMAD_DIR, _parse_cremad),
]


def build_metadata(verbose: bool = True) -> pd.DataFrame:
    """Scan the four dataset folders recursively and return the fused metadata.

    Raises a helpful error if no audio is found so the coding agent knows the
    datasets still need to be downloaded into ``data/``.
    """
    rows = []
    for corpus, root, parser in _CORPORA:
        files = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
        kept = 0
        for f in files:
            emotion = parser(f)
            if emotion in config.EMOTION_TO_ID:
                rows.append({"path": os.path.abspath(f),
                             "emotion": emotion,
                             "corpus": corpus})
                kept += 1
        if verbose:
            print(f"[data] {corpus:8s}: found {len(files):5d} wav files, "
                  f"kept {kept:5d} labelled samples (root: {root})")

    if not rows:
        raise FileNotFoundError(
            "No audio files found. Download RAVDESS, TESS, SAVEE and CREMA-D "
            "and place them under the folders configured in config.py "
            f"(DATA_DIR='{config.DATA_DIR}'). See README.md for links and the "
            "expected directory layout."
        )

    df = pd.DataFrame(rows)
    if verbose:
        print(f"[data] combined dataset: {len(df)} samples")
        print(df.groupby(['emotion']).size().to_string())
    return df


def split_metadata(df: pd.DataFrame):
    """Stratified 72 : 8 : 20 train/val/test split (before augmentation)."""
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df, test_size=config.TEST_FRACTION,
        stratify=df["emotion"], random_state=config.RANDOM_SEED)
    train, val = train_test_split(
        train_val, test_size=config.VAL_FRACTION_OF_TRAINVAL,
        stratify=train_val["emotion"], random_state=config.RANDOM_SEED)

    print(f"[split] train={len(train)}  val={len(val)}  test={len(test)} "
          f"(target ratios 72:8:20)")
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


if __name__ == "__main__":
    meta = build_metadata()
    split_metadata(meta)
```

## C.4 `augmentation.py`

EAAA emotion-aware policy (Novelty 2) plus the uniform base-paper baseline.

```python
"""Waveform augmentation: uniform (base-paper style) and EAAA (Novelty 2).

EAAA = Emotion-Aware Adaptive Augmentation. Instead of applying the same
techniques to every class, each training sample receives an augmentation
chosen to mirror the natural acoustic variability of its emotion class
(Table 2 of the proposal):

    happy    -> pitch shifting (+/- 2 semitones)
    sad      -> time stretching (0.8 - 1.2x)
    angry    -> noise injection (sigma 0.001 - 0.005 of peak amplitude)
    fear     -> time shifting (+/- 0.2 s) + light noise
    surprise -> time shifting (+/- 0.2 s)
    disgust  -> mild pitch shift (+/- 1 semitone) + mild stretch (0.9 - 1.1x)
    neutral  -> light noise injection only

All functions take and return a 1-D float32 waveform. Length is NOT
guaranteed to be preserved (time stretching changes it); callers must re-fix
the length afterwards (features.fix_length does this).
"""
from __future__ import annotations

import numpy as np
import librosa

import config


# ---------------------------------------------------------------------------
# Primitive augmentations
# ---------------------------------------------------------------------------
def add_noise(y: np.ndarray, sigma_lo: float = 0.001, sigma_hi: float = 0.005,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Additive Gaussian noise, sigma expressed relative to peak amplitude."""
    rng = rng or np.random.default_rng()
    sigma = rng.uniform(sigma_lo, sigma_hi) * max(np.max(np.abs(y)), 1e-8)
    return (y + rng.normal(0.0, sigma, size=y.shape)).astype(np.float32)


def time_shift(y: np.ndarray, max_shift_s: float = 0.2,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Circularly shift the waveform by up to +/- max_shift_s seconds."""
    rng = rng or np.random.default_rng()
    shift = int(rng.uniform(-max_shift_s, max_shift_s) * config.SAMPLE_RATE)
    return np.roll(y, shift).astype(np.float32)


def pitch_shift(y: np.ndarray, lo: float = -2.0, hi: float = 2.0,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Pitch shift by a random number of semitones in [lo, hi]."""
    rng = rng or np.random.default_rng()
    steps = rng.uniform(lo, hi)
    return librosa.effects.pitch_shift(
        y=y, sr=config.SAMPLE_RATE, n_steps=steps).astype(np.float32)


def time_stretch(y: np.ndarray, lo: float = 0.8, hi: float = 1.2,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Time stretch by a random rate in [lo, hi] (length changes!)."""
    rng = rng or np.random.default_rng()
    rate = rng.uniform(lo, hi)
    return librosa.effects.time_stretch(y=y, rate=rate).astype(np.float32)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def _aug_happy(y, rng):    return pitch_shift(y, -2.0, 2.0, rng)
def _aug_sad(y, rng):      return time_stretch(y, 0.8, 1.2, rng)
def _aug_angry(y, rng):    return add_noise(y, 0.001, 0.005, rng)
def _aug_fear(y, rng):     return add_noise(time_shift(y, 0.2, rng), 0.001, 0.003, rng)
def _aug_surprise(y, rng): return time_shift(y, 0.2, rng)
def _aug_disgust(y, rng):  return time_stretch(pitch_shift(y, -1.0, 1.0, rng), 0.9, 1.1, rng)
def _aug_neutral(y, rng):  return add_noise(y, 0.001, 0.003, rng)

# Emotion -> augmentation function (Table 2 of the proposal)
EAAA_POLICY = {
    "happy": _aug_happy,
    "sad": _aug_sad,
    "angry": _aug_angry,
    "fear": _aug_fear,
    "surprise": _aug_surprise,
    "disgust": _aug_disgust,
    "neutral": _aug_neutral,
}

# Base-paper style pool: one of the four techniques picked uniformly at
# random, regardless of the emotion class.
_UNIFORM_POOL = [
    lambda y, rng: add_noise(y, 0.001, 0.005, rng),
    lambda y, rng: time_shift(y, 0.2, rng),
    lambda y, rng: pitch_shift(y, -2.0, 2.0, rng),
    lambda y, rng: time_stretch(y, 0.8, 1.2, rng),
]


def augment(y: np.ndarray, emotion: str, emotion_aware: bool,
            rng: np.random.Generator) -> np.ndarray:
    """Apply one augmentation to waveform ``y``.

    emotion_aware=True  -> EAAA policy (Novelty 2)
    emotion_aware=False -> uniform base-paper policy (ablation baseline)
    """
    if emotion_aware:
        return EAAA_POLICY[emotion](y, rng)
    fn = _UNIFORM_POOL[rng.integers(0, len(_UNIFORM_POOL))]
    return fn(y, rng)


def plan_augmentation(train_df, emotion_aware: bool, seed: int | None = None):
    """Return the training item list: originals + planned augmented copies.

    Each item is a dict {path, emotion, augment(bool), emotion_aware(bool),
    seed(int)}. The per-item seed makes augmentation reproducible while still
    varying between items.

    The number of augmented copies is chosen so that the final training set
    size matches config.TARGET_TRAIN_SIZE (the base paper's expansion budget),
    or doubles the training set when TARGET_TRAIN_SIZE is None.
    """
    seed = config.RANDOM_SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    items = [{"path": r.path, "emotion": r.emotion, "augment": False,
              "emotion_aware": emotion_aware, "seed": 0}
             for r in train_df.itertuples()]

    n_train = len(items)
    if config.TARGET_TRAIN_SIZE is None:
        n_extra = n_train
    else:
        n_extra = max(config.TARGET_TRAIN_SIZE - n_train, 0)

    # sample source rows for the augmented copies (with replacement if needed)
    replace = n_extra > n_train
    chosen = rng.choice(n_train, size=n_extra, replace=replace)
    for i, idx in enumerate(chosen):
        src = items[idx]
        items.append({"path": src["path"], "emotion": src["emotion"],
                      "augment": True, "emotion_aware": emotion_aware,
                      "seed": int(seed + 1000 + i)})

    mode = "EAAA (emotion-aware)" if emotion_aware else "uniform (base paper)"
    print(f"[aug] policy={mode}  originals={n_train}  augmented={n_extra}  "
          f"total={len(items)}")
    return items
```

## C.5 `features.py`

Audio preprocessing and MFCC/ZCR/RMSE stream extraction with on-disk caching.

```python
"""Audio loading, preprocessing and hand-crafted feature extraction.

For every utterance three frame-level feature streams are computed
(base-paper feature set):

    MFCC : config.N_MFCC coefficients per frame, flattened  -> MFCC_LEN values
    ZCR  : zero-crossing rate per frame                     -> ZCR_LEN values
    RMSE : root-mean-square energy per frame                -> RMSE_LEN values

The streams are kept SEPARATE (not concatenated here) because the AFW module
(Novelty 1) needs to weight each stream individually before fusion inside the
model. With the default config the fused length is 2160+108+108 = 2376,
matching the base paper's (2376, 1) input.
"""
from __future__ import annotations

import os
import hashlib

import numpy as np
import librosa
from tqdm import tqdm

import config
from augmentation import augment


# ---------------------------------------------------------------------------
# Loading / preprocessing
# ---------------------------------------------------------------------------
def load_waveform(path: str) -> np.ndarray:
    """Load audio: resample to SAMPLE_RATE, skip OFFSET s, keep DURATION s,
    peak-normalise to [-1, 1] and pad/truncate to a fixed length."""
    y, _ = librosa.load(path, sr=config.SAMPLE_RATE,
                        offset=config.OFFSET, duration=config.DURATION)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return fix_length(y)


def fix_length(y: np.ndarray, n: int | None = None) -> np.ndarray:
    """Zero-pad or truncate a waveform to exactly ``n`` samples."""
    n = n or config.N_SAMPLES
    if len(y) >= n:
        return y[:n].astype(np.float32)
    return np.pad(y, (0, n - len(y))).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature streams
# ---------------------------------------------------------------------------
def _fix_frames(x: np.ndarray, n_frames: int) -> np.ndarray:
    """Pad/truncate the frame axis (last axis) to exactly n_frames."""
    if x.shape[-1] >= n_frames:
        return x[..., :n_frames]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, n_frames - x.shape[-1])]
    return np.pad(x, pad)


def extract_streams(y: np.ndarray):
    """Return (mfcc_flat, zcr, rmse) as fixed-length float32 vectors."""
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH)
    rmse = librosa.feature.rms(
        y=y, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH)
    mfcc = librosa.feature.mfcc(
        y=y, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
        n_fft=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH)

    zcr = _fix_frames(zcr.squeeze(0), config.N_FRAMES)
    rmse = _fix_frames(rmse.squeeze(0), config.N_FRAMES)
    mfcc = _fix_frames(mfcc, config.N_FRAMES).T.ravel()   # (frames*n_mfcc,)

    return (mfcc.astype(np.float32),
            zcr.astype(np.float32),
            rmse.astype(np.float32))


# ---------------------------------------------------------------------------
# Batch extraction with on-disk caching
# ---------------------------------------------------------------------------
def _items_fingerprint(items) -> str:
    h = hashlib.md5()
    for it in items:
        h.update(f"{it['path']}|{it['emotion']}|{it['augment']}|"
                 f"{it.get('emotion_aware')}|{it.get('seed')}".encode())
    h.update(f"{config.SAMPLE_RATE}|{config.DURATION}|{config.OFFSET}|"
             f"{config.FRAME_LENGTH}|{config.HOP_LENGTH}|{config.N_MFCC}"
             .encode())
    return h.hexdigest()[:16]


def build_feature_matrix(items, desc: str, use_cache: bool = True):
    """Extract features for a list of items (see augmentation.plan_augmentation).

    Returns a dict with keys:
        mfcc  (N, MFCC_LEN)   zcr (N, ZCR_LEN)   rmse (N, RMSE_LEN)
        y     (N,) integer class ids
    Results are cached in config.CACHE_DIR keyed by an md5 fingerprint of the
    item list + feature settings, so re-runs are instant.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        config.CACHE_DIR, f"{desc}_{_items_fingerprint(items)}.npz")

    if use_cache and os.path.exists(cache_path):
        print(f"[features] cache hit -> {cache_path}")
        z = np.load(cache_path)
        return {k: z[k] for k in ("mfcc", "zcr", "rmse", "y")}

    mfccs, zcrs, rmses, ys = [], [], [], []
    for it in tqdm(items, desc=f"[features] {desc}", unit="clip"):
        y_wave = load_waveform(it["path"])
        if it["augment"]:
            rng = np.random.default_rng(it["seed"])
            y_wave = fix_length(
                augment(y_wave, it["emotion"], it["emotion_aware"], rng))
        m, z, r = extract_streams(y_wave)
        mfccs.append(m); zcrs.append(z); rmses.append(r)
        ys.append(config.EMOTION_TO_ID[it["emotion"]])

    out = {"mfcc": np.stack(mfccs), "zcr": np.stack(zcrs),
           "rmse": np.stack(rmses), "y": np.asarray(ys, dtype=np.int64)}
    np.savez_compressed(cache_path, **out)
    print(f"[features] cached -> {cache_path}  "
          f"(mfcc {out['mfcc'].shape}, zcr {out['zcr'].shape}, "
          f"rmse {out['rmse'].shape})")
    return out


def df_to_items(df):
    """Convert a metadata DataFrame (val/test) into non-augmented items."""
    return [{"path": r.path, "emotion": r.emotion, "augment": False,
             "emotion_aware": False, "seed": 0} for r in df.itertuples()]
```

## C.6 `model.py`

AFW gating layer (Novelty 1), MSTC multi-scale block (Novelty 3), and the five-stage Conv1D backbone.

```python
"""Model architecture.

Implements:
  * AdaptiveFeatureWeighting (AFW, Novelty 1) - a small attention-style gate
    that learns per-sample softmax importance weights over the three feature
    streams (MFCC / ZCR / RMSE) and rescales each stream before fusion.
  * Multi-Scale Temporal Convolution (MSTC, Novelty 3) - the first Conv1D
    stage is replaced by three parallel branches with kernel sizes 3/5/7 whose
    outputs are concatenated channel-wise. The total filter budget of the
    stage is SPLIT across the branches, keeping parameters ~neutral.
  * The base-paper backbone - five Conv1D stages (512-512-256-256-128) with
    batch normalisation and max pooling, a Dense(512) head and 7-way Softmax.

`build_model` assembles any combination via `use_afw` / `use_mstc` flags,
giving the ablation-study structure for free.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

import config


class AdaptiveFeatureWeighting(layers.Layer):
    """Per-sample softmax gate over the three feature streams (Novelty 1).

    Input : list of 3 tensors  [(B, MFCC_LEN), (B, ZCR_LEN), (B, RMSE_LEN)]
    Output: (fused (B, INPUT_LEN), weights (B, 3))

    Each stream is summarised by its mean and standard deviation; a compact
    dense layer followed by a softmax produces one importance weight per
    stream. Weights are multiplied by 3 before scaling so that the "equal
    importance" solution (1/3, 1/3, 1/3) leaves stream magnitudes unchanged,
    which stabilises early training. Adds only a few thousand parameters.
    """

    def __init__(self, hidden_units: int = config.AFW_HIDDEN_UNITS, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.hidden = layers.Dense(hidden_units, activation="relu",
                                   name="afw_hidden")
        self.gate = layers.Dense(3, activation="softmax", name="afw_gate")

    def call(self, streams):
        summaries = []
        for s in streams:
            mean = tf.reduce_mean(s, axis=1, keepdims=True)
            std = tf.math.reduce_std(s, axis=1, keepdims=True)
            summaries.append(tf.concat([mean, std], axis=1))
        h = self.hidden(tf.concat(summaries, axis=1))     # (B, hidden)
        w = self.gate(h)                                  # (B, 3), sums to 1
        scaled = [s * (3.0 * w[:, i:i + 1]) for i, s in enumerate(streams)]
        fused = tf.concat(scaled, axis=1)                 # (B, INPUT_LEN)
        return fused, w

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg


def _mstc_block(x, total_filters: int):
    """Multi-scale first stage (Novelty 3): parallel kernels 3/5/7, filter
    budget split across branches so parameters stay ~equal to the original
    single-scale stage."""
    k = len(config.MSTC_KERNELS)
    base = total_filters // k
    branch_filters = [base] * k
    branch_filters[0] += total_filters - base * k   # absorb the remainder
    branches = []
    for ks, f in zip(config.MSTC_KERNELS, branch_filters):
        b = layers.Conv1D(f, kernel_size=ks, padding="same",
                          activation="relu", name=f"mstc_k{ks}")(x)
        branches.append(b)
    x = layers.Concatenate(axis=-1, name="mstc_concat")(branches)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(config.DROPOUT_CONV)(x)
    return x


def _conv_block(x, filters: int, name: str):
    """Standard base-paper Conv1D stage: Conv -> BN -> MaxPool -> Dropout."""
    x = layers.Conv1D(filters, kernel_size=config.BASE_KERNEL_SIZE,
                      padding="same", activation="relu", name=name)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(config.DROPOUT_CONV)(x)
    return x


def build_model(use_afw: bool, use_mstc: bool):
    """Build the SER model.

    Returns
    -------
    model        : tf.keras.Model mapping the three streams -> 7-way softmax
    weight_model : tf.keras.Model producing the AFW weights (or None when
                   use_afw=False); used for the interpretability analysis.
    """
    in_mfcc = layers.Input(shape=(config.MFCC_LEN,), name="mfcc")
    in_zcr = layers.Input(shape=(config.ZCR_LEN,), name="zcr")
    in_rmse = layers.Input(shape=(config.RMSE_LEN,), name="rmse")
    streams = [in_mfcc, in_zcr, in_rmse]

    weights_tensor = None
    if use_afw:
        fused, weights_tensor = AdaptiveFeatureWeighting(name="afw")(streams)
    else:
        fused = layers.Concatenate(axis=1, name="static_concat")(streams)

    x = layers.Reshape((config.INPUT_LEN, 1), name="to_sequence")(fused)

    # ---- five-stage Conv1D backbone -------------------------------------
    first, *rest = config.CONV_FILTERS
    if use_mstc:
        x = _mstc_block(x, first)
    else:
        x = _conv_block(x, first, name="conv1")
    for i, f in enumerate(rest, start=2):
        x = _conv_block(x, f, name=f"conv{i}")

    # ---- classification head --------------------------------------------
    x = layers.Flatten()(x)
    x = layers.Dense(config.DENSE_UNITS, activation="relu", name="dense")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_DENSE)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax",
                       name="softmax")(x)

    model = models.Model(streams, out, name="ser_afw_mstc_1dcnn")

    weight_model = None
    if use_afw:
        weight_model = models.Model(streams, weights_tensor,
                                    name="afw_weight_extractor")
    return model, weight_model


if __name__ == "__main__":
    m, _ = build_model(use_afw=True, use_mstc=True)
    m.summary()
```

## C.7 `losses.py`

CADL composite loss (Novelty 4): focal cross-entropy + pairwise confusion penalty.

```python
"""Confusion-Aware Discriminative Loss (CADL, Novelty 4).

Composite objective =
    focal cross-entropy                          (concentrates on hard samples)
  + lambda * pairwise confusion penalty          (extra margin cost on the
                                                  empirically confusable pairs
                                                  sad<->neutral, angry<->fear)

The penalty term for a pair (a, b) is
    y_a * (-log(1 - p_b))  +  y_b * (-log(1 - p_a))
i.e. whenever the true class is one member of a confusable pair, probability
mass assigned to its confusion partner is explicitly punished. Affects
training only - inference cost is unchanged.
"""
from __future__ import annotations

import tensorflow as tf

import config

_EPS = 1e-7


def _pair_indices():
    return [(config.EMOTION_TO_ID[a], config.EMOTION_TO_ID[b])
            for a, b in config.CONFUSION_PAIRS]


def cadl_loss(gamma: float = config.FOCAL_GAMMA,
              pair_lambda: float = config.PAIR_LAMBDA,
              use_focal: bool = True,
              use_pairwise: bool = True):
    """Build the CADL loss function (Keras-compatible closure).

    With use_focal=False and use_pairwise=False this reduces exactly to plain
    categorical cross-entropy, which keeps the ablation comparison clean.
    """
    pairs = _pair_indices()

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, _EPS, 1.0 - _EPS)

        # -- (focal) cross-entropy ------------------------------------------
        ce = -y_true * tf.math.log(y_pred)                    # (B, C)
        if use_focal:
            ce = tf.pow(1.0 - y_pred, gamma) * ce             # focal modulation
        total = tf.reduce_sum(ce, axis=-1)                    # (B,)

        # -- pairwise confusion penalty -------------------------------------
        if use_pairwise and pair_lambda > 0:
            pen = tf.zeros_like(total)
            for a, b in pairs:
                pen += y_true[:, a] * (-tf.math.log(1.0 - y_pred[:, b]))
                pen += y_true[:, b] * (-tf.math.log(1.0 - y_pred[:, a]))
            total = total + pair_lambda * pen

        return total

    loss.__name__ = "cadl_loss"
    return loss


def get_loss(use_cadl: bool):
    """Return CADL when the novelty is enabled, else plain cross-entropy."""
    if use_cadl:
        return cadl_loss()
    return "categorical_crossentropy"
```

## C.8 `utils.py`

Reproducibility seeding, per-stream standardisation, and plotting helpers.

```python
"""Shared utilities: reproducibility, per-stream scaling, plotting."""
from __future__ import annotations

import os
import json
import random

import numpy as np

import config


def set_seed(seed: int = config.RANDOM_SEED):
    """Best-effort determinism across python / numpy / tensorflow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Per-stream standardisation (fit on train, applied to val/test)
# ---------------------------------------------------------------------------
class StreamScalers:
    """StandardScaler per feature stream, since MFCC / ZCR / RMSE live on very
    different numeric ranges and must not share statistics."""

    STREAMS = ("mfcc", "zcr", "rmse")

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scalers = {s: StandardScaler() for s in self.STREAMS}

    def fit(self, feats: dict):
        for s in self.STREAMS:
            self.scalers[s].fit(feats[s])
        return self

    def transform(self, feats: dict) -> list:
        """Return model inputs in the order expected by build_model."""
        return [self.scalers[s].transform(feats[s]).astype(np.float32)
                for s in self.STREAMS]

    def save(self, path: str):
        import joblib
        joblib.dump(self.scalers, path)

    @classmethod
    def load(cls, path: str):
        import joblib
        obj = cls.__new__(cls)
        obj.scalers = joblib.load(path)
        return obj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_history(history: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_path: str, normalise: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype(float)
    if normalise:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
    ax.set_xticks(range(config.NUM_CLASSES), config.EMOTIONS, rotation=45)
    ax.set_yticks(range(config.NUM_CLASSES), config.EMOTIONS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            label = f"{cm[i, j]}" if not normalise else f"{data[i, j]:.2f}"
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if data[i, j] > data.max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
```

## C.9 `evaluate.py`

Full base-paper metric suite, confusion-pair analysis, and AFW interpretability.

```python
"""Evaluation with the full base-paper metric suite.

Computes: accuracy (+95% CI), per-class and macro precision / recall / F1,
per-class specificity, G-mean, MCC, Cohen's Kappa, macro one-vs-rest AUC,
the confusion matrix, and the misclassification counts of the two targeted
confusion pairs (sad<->neutral, angry<->fear).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)

import config
from utils import plot_confusion_matrix, save_json


def _specificity_per_class(cm: np.ndarray) -> np.ndarray:
    """Specificity_c = TN_c / (TN_c + FP_c) from the multi-class CM."""
    total = cm.sum()
    spec = np.zeros(cm.shape[0])
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        spec[c] = tn / max(tn + fp, 1)
    return spec


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray,
                         out_dir: str | None = None, prefix: str = "test"):
    """Compute all metrics from integer labels and predicted probabilities.

    Returns the metrics dict; when out_dir is given also writes
    {prefix}_metrics.json, {prefix}_classification_report.csv and the
    confusion-matrix plots.
    """
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))

    acc = accuracy_score(y_true, y_pred)
    n = len(y_true)
    ci_half = 1.96 * np.sqrt(acc * (1 - acc) / n)          # normal-approx 95% CI

    report = classification_report(
        y_true, y_pred, labels=range(config.NUM_CLASSES),
        target_names=config.EMOTIONS, output_dict=True, zero_division=0)

    spec = _specificity_per_class(cm)
    recall_per_class = np.array(
        [report[e]["recall"] for e in config.EMOTIONS])
    gmean_per_class = np.sqrt(np.clip(recall_per_class * spec, 0, None))

    y_true_1h = np.eye(config.NUM_CLASSES)[y_true]
    try:
        auc = roc_auc_score(y_true_1h, y_prob, multi_class="ovr",
                            average="macro")
    except ValueError:
        auc = float("nan")   # e.g. a class missing from y_true

    pair_confusions = {}
    for a, b in config.CONFUSION_PAIRS:
        ia, ib = config.EMOTION_TO_ID[a], config.EMOTION_TO_ID[b]
        pair_confusions[f"{a}<->{b}"] = int(cm[ia, ib] + cm[ib, ia])

    metrics = {
        "accuracy": acc,
        "accuracy_95ci": [acc - ci_half, acc + ci_half],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "specificity_per_class": {e: float(s) for e, s
                                  in zip(config.EMOTIONS, spec)},
        "macro_specificity": float(spec.mean()),
        "gmean_per_class": {e: float(g) for e, g
                            in zip(config.EMOTIONS, gmean_per_class)},
        "macro_gmean": float(gmean_per_class.mean()),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "auc_ovr_macro": auc,
        "confusion_pair_errors": pair_confusions,
        "n_test": int(n),
    }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        save_json(metrics, os.path.join(out_dir, f"{prefix}_metrics.json"))
        pd.DataFrame(report).T.to_csv(
            os.path.join(out_dir, f"{prefix}_classification_report.csv"))
        np.savetxt(os.path.join(out_dir, f"{prefix}_confusion_matrix.csv"),
                   cm, fmt="%d", delimiter=",")
        plot_confusion_matrix(cm, os.path.join(
            out_dir, f"{prefix}_confusion_matrix.png"), normalise=False)
        plot_confusion_matrix(cm, os.path.join(
            out_dir, f"{prefix}_confusion_matrix_norm.png"), normalise=True)

    print(f"[eval] accuracy={acc:.4f}  macroF1={metrics['macro_f1']:.4f}  "
          f"MCC={metrics['mcc']:.4f}  kappa={metrics['cohen_kappa']:.4f}  "
          f"AUC={metrics['auc_ovr_macro']:.4f}")
    print(f"[eval] confusion-pair errors: {pair_confusions}")
    return metrics


def afw_interpretability(weight_model, inputs, y_true: np.ndarray,
                         out_dir: str):
    """Novelty-1 interpretability: mean AFW stream weight per emotion class."""
    w = weight_model.predict(inputs, verbose=0)       # (N, 3)
    df = pd.DataFrame(w, columns=["mfcc", "zcr", "rmse"])
    df["emotion"] = [config.ID_TO_EMOTION[i] for i in y_true]
    table = df.groupby("emotion").mean().reindex(config.EMOTIONS)
    table.to_csv(os.path.join(out_dir, "afw_weights_per_emotion.csv"))
    print("[eval] AFW mean stream weights per emotion:")
    print(table.round(3).to_string())
    return table


if __name__ == "__main__":
    # Standalone re-evaluation of a finished run:
    #   python evaluate.py runs/<tag>
    import sys
    import tensorflow as tf
    from data_loader import build_metadata, split_metadata
    from features import build_feature_matrix, df_to_items
    from utils import StreamScalers
    from model import AdaptiveFeatureWeighting
    from losses import cadl_loss

    run_dir = sys.argv[1]
    meta = build_metadata()
    _, _, test_df = split_metadata(meta)
    test_feats = build_feature_matrix(df_to_items(test_df), desc="test")

    scalers = StreamScalers.load(os.path.join(run_dir, "scalers.joblib"))
    x_test = scalers.transform(test_feats)

    model = tf.keras.models.load_model(
        os.path.join(run_dir, "best_model.keras"),
        custom_objects={"AdaptiveFeatureWeighting": AdaptiveFeatureWeighting,
                        "cadl_loss": cadl_loss()},
        compile=False)
    y_prob = model.predict(x_test, batch_size=config.BATCH_SIZE, verbose=1)
    evaluate_predictions(test_feats["y"], y_prob, out_dir=run_dir)
```

## C.10 `train.py`

End-to-end training pipeline with per-novelty CLI flags.

```python
"""End-to-end training pipeline.

Runs the full workflow of Section 5.7 of the proposal:
  1. data fusion            (data_loader.build_metadata)
  2. preprocessing          (features.load_waveform)
  3. 72:8:20 split          (data_loader.split_metadata, BEFORE augmentation)
  4. EAAA / uniform aug.    (augmentation.plan_augmentation)   [Novelty 2]
  5. feature extraction     (features.build_feature_matrix)
  6. AFW weighting          (model.AdaptiveFeatureWeighting)   [Novelty 1]
  7. MSTC backbone          (model.build_model)                [Novelty 3]
  8. CADL training          (losses.cadl_loss)                 [Novelty 4]
  9. evaluation             (evaluate.evaluate_predictions)

Each novelty can be switched on/off from the CLI, e.g.:

    python train.py --tag full                       # all four novelties
    python train.py --tag base --no-afw --no-eaaa --no-mstc --no-cadl
    python train.py --tag afw_only --no-eaaa --no-mstc --no-cadl
"""
from __future__ import annotations

import os
import argparse

import numpy as np

import config
from data_loader import build_metadata, split_metadata
from augmentation import plan_augmentation
from features import build_feature_matrix, df_to_items
from utils import set_seed, StreamScalers, plot_history, save_json


def run_experiment(use_afw: bool, use_eaaa: bool, use_mstc: bool,
                   use_cadl: bool, tag: str, epochs: int = config.EPOCHS):
    """Train one configuration and return its test metrics dict."""
    import tensorflow as tf
    from model import build_model
    from losses import get_loss
    from evaluate import evaluate_predictions, afw_interpretability

    set_seed()
    run_dir = os.path.join(config.RUNS_DIR, tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== experiment '{tag}'  AFW={use_afw} EAAA={use_eaaa} "
          f"MSTC={use_mstc} CADL={use_cadl} ===")

    # ---- steps 1-3: fuse, split (before augmentation - no leakage) --------
    meta = build_metadata()
    train_df, val_df, test_df = split_metadata(meta)

    # ---- step 4: augmentation plan (training subset only) -----------------
    train_items = plan_augmentation(train_df, emotion_aware=use_eaaa)

    # ---- step 5: feature extraction (cached) ------------------------------
    aug_key = "eaaa" if use_eaaa else "uniform"
    train_feats = build_feature_matrix(train_items, desc=f"train_{aug_key}")
    val_feats = build_feature_matrix(df_to_items(val_df), desc="val")
    test_feats = build_feature_matrix(df_to_items(test_df), desc="test")

    # ---- per-stream standardisation (fit on train only) -------------------
    scalers = StreamScalers().fit(train_feats)
    scalers.save(os.path.join(run_dir, "scalers.joblib"))
    x_train = scalers.transform(train_feats)
    x_val = scalers.transform(val_feats)
    x_test = scalers.transform(test_feats)

    y_train = np.eye(config.NUM_CLASSES)[train_feats["y"]]
    y_val = np.eye(config.NUM_CLASSES)[val_feats["y"]]

    # ---- steps 6-7: model --------------------------------------------------
    model, weight_model = build_model(use_afw=use_afw, use_mstc=use_mstc)
    model.summary(line_length=100)

    # ---- step 8: compile + train ------------------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss=get_loss(use_cadl),
        metrics=["accuracy"])

    ckpt_path = os.path.join(run_dir, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True,
            verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE, min_lr=config.MIN_LR,
            verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "training_log.csv")),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=config.BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        verbose=2)

    plot_history(history.history, os.path.join(run_dir, "training_curves.png"))

    # ---- step 9: evaluation ------------------------------------------------
    y_prob = model.predict(x_test, batch_size=config.BATCH_SIZE, verbose=0)
    metrics = evaluate_predictions(test_feats["y"], y_prob, out_dir=run_dir)
    metrics["config"] = {"use_afw": use_afw, "use_eaaa": use_eaaa,
                         "use_mstc": use_mstc, "use_cadl": use_cadl,
                         "params": int(model.count_params())}
    save_json(metrics, os.path.join(run_dir, "test_metrics.json"))

    # ---- Novelty-1 interpretability ---------------------------------------
    if use_afw and weight_model is not None:
        afw_interpretability(weight_model, x_test, test_feats["y"], run_dir)

    print(f"[done] artefacts saved under {run_dir}/")
    return metrics


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="full", help="run name under runs/")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    for name in ("afw", "eaaa", "mstc", "cadl"):
        p.add_argument(f"--{name}", dest=name, action="store_true",
                       default=True)
        p.add_argument(f"--no-{name}", dest=name, action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(use_afw=args.afw, use_eaaa=args.eaaa,
                   use_mstc=args.mstc, use_cadl=args.cadl,
                   tag=args.tag, epochs=args.epochs)
```

## C.11 `ablation.py`

Six-configuration ablation study runner.

```python
"""Component-wise ablation study (Section 5.6 / Expected Outcome).

Trains six configurations and writes a comparison table:

    base       : no novelties (uniform augmentation, plain cross-entropy)
    +AFW       : base + Adaptive Feature Weighting
    +EAAA      : base + Emotion-Aware Adaptive Augmentation
    +MSTC      : base + Multi-Scale Temporal Convolution
    +CADL      : base + Confusion-Aware Discriminative Loss
    all        : all four novelties combined (the proposed model)

Run:  python ablation.py            (full 50-epoch runs)
      python ablation.py --epochs 15  (quick sanity sweep)
"""
from __future__ import annotations

import os
import argparse

import pandas as pd

import config
from train import run_experiment

CONFIGS = [
    ("base",  dict(use_afw=False, use_eaaa=False, use_mstc=False, use_cadl=False)),
    ("afw",   dict(use_afw=True,  use_eaaa=False, use_mstc=False, use_cadl=False)),
    ("eaaa",  dict(use_afw=False, use_eaaa=True,  use_mstc=False, use_cadl=False)),
    ("mstc",  dict(use_afw=False, use_eaaa=False, use_mstc=True,  use_cadl=False)),
    ("cadl",  dict(use_afw=False, use_eaaa=False, use_mstc=False, use_cadl=True)),
    ("all",   dict(use_afw=True,  use_eaaa=True,  use_mstc=True,  use_cadl=True)),
]


def main(epochs: int):
    rows = []
    for tag, cfg in CONFIGS:
        metrics = run_experiment(tag=f"ablation_{tag}", epochs=epochs, **cfg)
        rows.append({
            "config": tag,
            **{k: v for k, v in cfg.items()},
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mcc": metrics["mcc"],
            "cohen_kappa": metrics["cohen_kappa"],
            "auc": metrics["auc_ovr_macro"],
            "sad<->neutral_errors":
                metrics["confusion_pair_errors"].get("sad<->neutral"),
            "angry<->fear_errors":
                metrics["confusion_pair_errors"].get("angry<->fear"),
            "params": metrics["config"]["params"],
        })
        # incremental save so partial sweeps are never lost
        os.makedirs(config.RUNS_DIR, exist_ok=True)
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(config.RUNS_DIR, "ablation_results.csv"),
                     index=False)
        print("\n==== ablation table so far ====")
        print(table.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = p.parse_args()
    main(args.epochs)
```

## C.12 `README.md`

Repository quickstart.

```markdown
# Adaptive Feature-Weighted 1D-CNN with Emotion-Aware Augmentation for Robust SER

Implementation of the project proposal "Adaptive Feature-Weighted 1D-CNN with
Emotion-Aware Augmentation for Robust Speech Emotion Recognition" (base paper:
Chourasia et al., *Scientific Reports*, 2026). Full specification lives in
`PROJECT_SPEC.md` — read that first.

## Quick start

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. datasets — download and place under data/ (see PROJECT_SPEC.md, Part B.2)
#    data/RAVDESS   data/TESS   data/SAVEE   data/CREMA-D

# 3. sanity check the dataset scan (should report ~12,162 samples)
python data_loader.py

# 4. train the full proposed model (all four novelties)
python train.py --tag full

# 5. train the base-paper reproduction (no novelties)
python train.py --tag base --no-afw --no-eaaa --no-mstc --no-cadl

# 6. run the complete 6-way ablation study
python ablation.py
```

Artefacts (best model, metrics JSON, classification report, confusion
matrices, training curves, AFW interpretability table) are written to
`runs/<tag>/`. The ablation summary lands in `runs/ablation_results.csv`.

## Repository layout

```
config.py        all hyperparameters, paths, emotion classes, novelty settings
data_loader.py   corpus scanning + label parsing (RAVDESS/TESS/SAVEE/CREMA-D)
augmentation.py  EAAA policy (Novelty 2) + uniform baseline augmentation
features.py      MFCC/ZCR/RMSE stream extraction with on-disk caching
model.py         AFW gate (Novelty 1), MSTC block (Novelty 3), Conv1D backbone
losses.py        CADL composite loss (Novelty 4)
train.py         end-to-end pipeline with per-novelty CLI flags
evaluate.py      full metric suite + confusion analysis + AFW interpretability
ablation.py      6-configuration ablation runner
```
```

---

*End of specification. Start with Part B.5 once the datasets from B.2 are in place.*
