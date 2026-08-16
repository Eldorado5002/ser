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
