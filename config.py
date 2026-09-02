"""Central configuration for the Meltwater Hangman pipeline.

All important hyper-parameters live here so the notebook and the local
modules share a single source of truth.  Nothing in the pipeline makes an
internet / pretrained-weights / external-API call: the model is built from
scratch with plain PyTorch and a self-contained character tokeniser.
"""
import os


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
TEST_FILE = os.path.join(DATA_DIR, "test.txt")
SAMPLE_SUBMISSION = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_SPLIT_CSV = os.path.join(DATA_DIR, "train_split.csv")
VAL_SPLIT_CSV = os.path.join(DATA_DIR, "val_split.csv")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 1920

# ---------------------------------------------------------------------------
# Train / validation split (word-level, performed BEFORE state generation)
# ---------------------------------------------------------------------------
VAL_SPLIT_RATIO = 0.10
FINAL_FULL_DATA = False  # switch to True to train on all 225300 words

# ---------------------------------------------------------------------------
# Character-level alphabet & token scheme
# Vocabulary is a-z only.  We add a few game-state machinery tokens which are
# NOT part of the word vocabulary and never appear in a hidden word.
# ---------------------------------------------------------------------------
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
A_CODE = ord("a")
NUM_LETTERS = 26  # model outputs 26 logits (one per a-z)

PAD = 26  # padding token id
MASK = 27  # unrevealed letter position
CLS = 28  # classification / pool token
SEP = 29  # separator between guessed-letters and the masked word
VOCAB_SIZE = 30  # 26 letters + PAD + MASK + CLS + SEP

# ---------------------------------------------------------------------------
# Model dimensions (CANINE-S style character transformer)
# ---------------------------------------------------------------------------
MODEL_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
FF_DIM = 1024
DROPOUT = 0.1
MAX_SEQ_LEN = 64  # max tokens per input (CLS + 26 guesses + SEP + word + pad)
MAX_WORD_LEN = 32  # max hidden-word length supported (<=36 keeps layout within MAX_SEQ_LEN)

# Fixed token layout (length = 1 + 26 + 1 + MAX_WORD_LEN = 60 <= 64):
#   idx 0            : CLS (pool token)
#   idx 1..26        : guessed letters in guess order, PAD if slot unused
#   idx 27           : SEP
#   idx 28..28+W-1   : word letters; MASK if hidden, letter-id if revealed
#   idx 28+W..59     : PAD
LAYOUT_GUESS_START = 1
LAYOUT_GUESS_END = 27   # exclusive
LAYOUT_SEP = 27
LAYOUT_WORD_START = 28

# ---------------------------------------------------------------------------
# Training / optimisation
# ---------------------------------------------------------------------------
BATCH_SIZE = 512
VAL_BATCH_SIZE = 4096
GRAD_ACCUM_STEPS = 1
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM_STEPS

PHASE1_EPOCHS = 3
SELF_PLAY_ROUNDS = 2
SELF_PLAY_EPOCHS_PER_ROUND = 1

MAX_TRAIN_STEPS = None        # hard cap on optimizer steps per phase-epoch (None = all)
MAX_MC_WORDS_PER_EPOCH = 0    # 0 => use all train words to generate MC states

# Self-play
SELF_PLAY_BATCH = 4096        # batched rollout width
MC_MIX_RATIO = 0.30           # fraction of a self-play training batch kept as fresh MC states
MAX_WRONG_GUESSES = 6         # six-strike hangman

# Bias sampling
CORRECT_GUESS_PROB = 0.40     # probability that a Monte-Carlo guess is a correct letter

# ---------------------------------------------------------------------------
# Optimiser / scheduler
# ---------------------------------------------------------------------------
LR = 3e-4
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = "cosine"       # "cosine" or "constant" or "linear_warmup"
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0

# ---------------------------------------------------------------------------
# Mixed precision / device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda"
USE_AMP = True
# On a machine without CUDA torch will fall back automatically; the notebook
# sets DEVICE appropriately at runtime.
try:
    import torch as _torch
    if not _torch.cuda.is_available():
        DEVICE = "cpu"
        USE_AMP = False
except Exception:
    DEVICE = "cpu"
    USE_AMP = False

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
VAL_WORDS = None              # None => use full held-out split
VAL_AT_END_OF_PHASE1 = True
VAL_AT_END_OF_SELFPLAY = True

# ---------------------------------------------------------------------------
# Dataloader workers
# ---------------------------------------------------------------------------
NUM_WORKERS = 0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_INTERVAL = 200
