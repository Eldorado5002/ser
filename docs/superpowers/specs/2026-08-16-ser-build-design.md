# SER Project — Build Design

**Date:** 2026-08-16
**Status:** Approved (design), pending spec review
**Goal:** Take the existing, never-executed SER codebase to a verified, fully-run
research deliverable.

---

## 1. Context

The repository contains a complete implementation of the proposal *"Adaptive
Feature-Weighted 1D-CNN with Emotion-Aware Augmentation for Robust Speech
Emotion Recognition"* (base paper: Chourasia et al., *Scientific Reports*,
2026). Ten Python modules, a 1,313-line specification, and a proposal PDF.

**Nothing has ever been executed.** There is no `data/`, no `features_cache/`,
no `runs/`, and no installed dependency. The code is internally consistent but
entirely unproven.

### Decisions taken (user-confirmed)

| Decision | Choice |
|---|---|
| Definition of done | Full research deliverable — all runs, metrics, ablation, report artefacts |
| Compute | Kaggle free tier (~30 GPU-h/week; job needs ~6–9) |
| If target missed | Run honest first, decide together. **Test set stays sealed** — any tuning uses validation only |
| Verification strategy | Local test suite + Kaggle for training (Approach A) |
| CREMA-D source | `ejlok1/cremad` (audio-only) |

---

## 2. Verified dataset facts

The four corpora were inspected via the Kaggle API, and the **real parsers from
`data_loader.py` were run against the real filenames**. Label parsing depends
only on filenames, so the entire label pipeline was validated offline.

### Parser results — 100% parse rate

| Corpus | Kaggle slug | Canonical subdir | Parsed |
|---|---|---|---|
| RAVDESS | `uwrfkaggler/ravdess-emotional-speech-audio` | `audio_speech_actors_01-24/` | 1,440 / 1,440 |
| TESS | `ejlok1/toronto-emotional-speech-set-tess` | `TESS Toronto emotional speech set data/` | 2,800 / 2,800 |
| SAVEE | `ejlok1/surrey-audiovisual-expressed-emotion-savee` | `ALL/` | 480 / 480 |
| CREMA-D | `ejlok1/cremad` | `AudioWAV/` | 7,442 / 7,442 |
| | | **Total** | **12,162** ✓ |

Zero unparsed files. Matches the spec target exactly.

### Verified class distribution

| Emotion | Count |
|---|---|
| angry | 1,923 |
| disgust | 1,923 |
| fear | 1,923 |
| happy | 1,923 |
| neutral | 1,895 |
| sad | 1,923 |
| surprise | **652** |
| **Total** | **12,162** |

`neutral` includes RAVDESS `calm` (192) merged per spec B.2. `surprise` is the
minority class because CREMA-D contains none — this is the class EAAA is
expected to help most.

### CRITICAL: duplicate-file defect in two mirrors

| Corpus | .wav in dataset | Expected | Duplicated basenames |
|---|---|---|---|
| RAVDESS | **2,880** | 1,440 | **1,440 (all)** |
| TESS | **5,600** | 2,800 | **2,800 (all)** |
| SAVEE | 480 | 480 | 0 |
| CREMA-D | 7,442 | 7,442 | 0 |

- **RAVDESS** ships `Actor_01/`…`Actor_24/` *and* `audio_speech_actors_01-24/`
  — the same 1,440 clips twice.
- **TESS** ships `TESS Toronto emotional speech set data/` *and*
  `tess toronto emotional speech set data/` — identical content, differing only
  in capitalisation (distinct directories on Linux).

`build_metadata` globs recursively, so a naive scan yields **16,402 samples
instead of 12,162** without raising any error.

**Why this is fatal, not cosmetic.** The duplicates are byte-identical
recordings, and `split_metadata` splits randomly over utterances. A clip and its
twin land on opposite sides of the train/test boundary ~32% of the time
(`2 × 0.2 × 0.8`). That places an estimated **~1,357 of ~3,280 test samples —
about 41% — with an exact copy in the training set.** Reported accuracy would
measure memorisation, not generalisation, and would look plausible (likely 98%+)
while being worthless.

**Mitigations (defence in depth):**
1. Symlink only the canonical subdirectory of each corpus (table above).
2. A basename-collision guard in `build_metadata` that raises on duplicates.
3. Per-corpus count asserts before training begins.
4. A regression test fed the real filename listings.

Symlinks being correct must not be something we merely hope for.

---

## 3. What gets built

| Component | Purpose | Runs where |
|---|---|---|
| `.venv` (Python 3.10) | Local CPU TensorFlow, so correctness is verified not asserted | Local |
| `tests/` | The evidence behind "perfectly working" | Local, seconds |
| `report.py` | Aggregates all runs into report-ready tables; checks acceptance criteria | Local |
| `notebooks/01_features.ipynb` | Dataset validation + feature extraction (CPU session, free) | Kaggle |
| `notebooks/02_train.ipynb` | Resumable 6-config training runner | Kaggle GPU |

The ten existing modules are **proven, not rewritten**. Only the changes in
§5 are made.

---

## 4. Test suite

Design rule: **if the report asserts it, a test proves it.**

| Test file | Proves | Data needed |
|---|---|---|
| `test_data_loader.py` | All four filename conventions; `sa`/`su` two-letter precedence; RAVDESS `calm→neutral`; CREMA-D's absent surprise | Filenames only |
| `test_no_duplicates.py` | Basename-collision guard fires on the real RAVDESS/TESS layouts; canonical subdirs yield exactly 12,162 | Real filename listings (cached JSON) |
| `test_split.py` | 72:8:20 ratios; stratified; **zero path overlap** train↔test | None |
| `test_augmentation.py` | EAAA applies the Table 2 policy per class; time-stretched waveforms restored to exactly `N_SAMPLES` | Synthetic |
| `test_features.py` | Fused length exactly **2376**; cache fingerprint invalidates on settings change | Synthetic |
| `test_model.py` | AFW weights sum to 1 and sit near ⅓ at init; **MSTC parameter-neutral**; all 4 flag combos build | Synthetic |
| `test_losses.py` | CADL collapses **exactly** to categorical cross-entropy when both terms off; pairwise penalty punishes the confusion partner | None |
| `test_train_integration.py` | 2-epoch run on synthetic data; reloading the saved model reproduces reported metrics | Synthetic |

### Why the equivalence tests matter most

`test_losses.py` (CADL ≡ CE when disabled) and `test_model.py` (MSTC parameter
neutrality) are what make the **ablation table interpretable**. Without them, a
delta between `base` and `+CADL` could be a loss-scaling artefact rather than
the novelty, and a delta between `base` and `+MSTC` could be extra capacity
rather than multi-scale receptive fields. The ablation is the headline
contribution — these two tests are what let it be defended.

`pytest` is added as a dev dependency.

---

## 5. Changes to existing code

1. **Checkpoint/eval consistency** (`train.py:83-88`). `ModelCheckpoint`
   monitors `val_accuracy` while `EarlyStopping(restore_best_weights=True)`
   monitors `val_loss`, and the reported metrics come from the restored model,
   not the saved file. Re-running `evaluate.py` on the same run can print
   different numbers than `test_metrics.json`. Fix: monitor one quantity in
   both, so `best_model.keras` *is* the model the metrics describe.

2. **Duplicate guard + per-corpus asserts** (`data_loader.py`). Raise on
   duplicate basenames within a corpus; expose expected per-corpus counts.

3. **Testability seam** (`train.py`). `run_experiment` calls `build_metadata()`
   internally, hard-requiring the real corpora. Add an optional
   `metadata=None` parameter so the integration test can inject a small
   synthetic DataFrame. Production behaviour unchanged when omitted.

4. **`report.py`** (new). Nothing currently collects `base` and `full`
   alongside the six ablation runs. Produces the base-vs-full comparison and
   machine-checks the §7 acceptance criteria.

---

## 6. Kaggle execution

### Session 1 — CPU (consumes **zero** GPU quota)

1. Attach the four datasets.
2. Symlink canonical subdirs into one **pinned** root.
3. Assert per-corpus counts (1,440 / 2,800 / 480 / 7,442 = 12,162); fail loudly.
4. Extract **four** caches: `train_eaaa`, `train_uniform`, `val`, `test`.
   Both augmentation policies are required — the ablation runs EAAA and
   non-EAAA configs.
5. Save `features_cache/` as notebook output → publish as a Kaggle Dataset.

### Session 2+ — GPU

1. Attach the corpora, the feature-cache dataset, and any previous `runs/`.
2. **Assert a cache hit before training starts.**
3. Run configs; save `runs/` as output.

### The pinned-path trap

The cache fingerprint (`features.py:83-91`) hashes **absolute file paths**. If
the symlink root differs by one character between the CPU and GPU sessions,
every cache silently misses and re-extracts — burning 2–3 GPU-hours doing CPU
work. The root is therefore pinned identically in both notebooks, and the
training notebook asserts a cache *hit* up front rather than discovering the
miss 20 minutes in.

### Resumability

6 runs × ~1–1.5 h can approach the 12-hour session cap. The runner skips any
config whose `test_metrics.json` already exists, so runs can be spread across
sessions by re-attaching the previous output. No babysitting; no lost work if a
session dies.

### The runs — 6 distinct configurations, not 8

The README presents `--tag base`, `--tag full`, and a 6-way ablation as separate
work, but they overlap:

- `train.py --tag base --no-afw --no-eaaa --no-mstc --no-cadl` is **identical**
  to the ablation's `base` config.
- `train.py --tag full` is **identical** to the ablation's `all` config.

Same flags, same seed, same data → same result. Running all eight would spend
~2–3 GPU-hours recomputing two runs.

Therefore **six runs** execute:

| Tag | AFW | EAAA | MSTC | CADL | Role |
|---|---|---|---|---|---|
| `base` | ✗ | ✗ | ✗ | ✗ | base-paper reproduction |
| `afw` | ✓ | ✗ | ✗ | ✗ | Novelty 1 isolated |
| `eaaa` | ✗ | ✓ | ✗ | ✗ | Novelty 2 isolated |
| `mstc` | ✗ | ✗ | ✓ | ✗ | Novelty 3 isolated |
| `cadl` | ✗ | ✗ | ✗ | ✓ | Novelty 4 isolated |
| `full` | ✓ | ✓ | ✓ | ✓ | proposed model |

`report.py` presents `base` and `full` in both the headline comparison and the
ablation table — one run, two roles. Estimated **~6–9 GPU-hours** total.

---

## 7. Acceptance criteria

From spec B.6, machine-checked by `report.py`:

- [ ] Scan reports **12,162** samples across exactly 7 classes
- [ ] No duplicate basenames within any corpus
- [ ] Split ≈ 8,756 / 973 / 2,433 (72:8:20), stratified, before augmentation
- [ ] `config.INPUT_LEN == 2376`
- [ ] EAAA applies the Table 2 policy; `--no-eaaa` falls back to the uniform pool
- [ ] `base` config lands ≈ 94–95% (base-paper reproduction)
- [ ] `full` config — **honest number, whatever it is**; target 95.5–96.5%
- [ ] `test_metrics.json` contains accuracy + 95% CI, macro P/R/F1, per-class
      specificity, G-mean, MCC, Cohen's Kappa, macro OvR AUC, and the
      sad↔neutral / angry↔fear error counts
- [ ] CADL runs show reduced confusion-pair errors vs base
- [ ] `ablation_results.csv` contains all six configs with parameter counts
- [ ] `afw_weights_per_emotion.csv` exists (interpretability deliverable)

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Kaggle phone verification incomplete | **Blocking** | User action; link is inside a notebook's Settings sidebar, not account settings |
| Duplicate-file leakage | **Critical** | Canonical symlinks + code guard + test (§2) |
| Cache fingerprint miss between sessions | High (wastes quota) | Pinned root + assert cache hit before training |
| `full` misses 95.5–96.5% | Medium | Honest-first policy; ablation stands on its own regardless |
| Session timeout mid-campaign | Low | Resumable runner |
| TF install friction on Windows | Low | Python 3.10; CPU-only build is the default on Windows |

### Known limitation (documented, not fixed)

The 72:8:20 split is **utterance-level, not speaker-independent**. TESS is 2
speakers × 200 words × 7 emotions, so the same speaker appears in train and
test, inflating accuracy relative to a speaker-independent protocol. This is
inherited from the base paper and is **retained deliberately** — changing it
would break comparability with the 94.91% figure this project exists to beat.
It should be stated as a limitation in the report.

---

## 9. Out of scope

- Speaker-independent splitting (see above)
- Architecture changes beyond the four specified novelties
- Hyperparameter tuning (revisited only after honest results, on validation)
- Real-time inference / deployment tooling
- Retraining the base paper's exact 40-MFCC / 3-second variant

---

## 10. Build order

```
git + .gitignore                       [done]
spec (this document)                   [done]
implementation plan
  ↓
local venv (py3.10 + CPU TensorFlow)
  ↓
tests/                                 ← MILESTONE 1: green, zero datasets
  ↓
train.py fixes + data_loader guard + report.py
  ↓
notebooks/01_features.ipynb            (Kaggle CPU, free)
  ↓
notebooks/02_train.ipynb               (Kaggle GPU, ~6-9 h, resumable)
  ↓
report.py → final tables               ← MILESTONE 2: deliverable
```

**Milestone 1** is the critical one: when the local suite is green, all four
novelties are proven correctly wired — AFW weights normalise, MSTC is
parameter-neutral, CADL collapses exactly to cross-entropy when disabled, no
split leakage, input length exactly 2376 — with **no audio downloaded**. That
is where "perfectly working" stops being a claim and becomes evidence.
