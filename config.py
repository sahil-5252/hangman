"""Central configuration for the Meltwater Hangman pipeline.

All important hyper-parameters live here so the notebook and the local
modules share a single source of truth.  Nothing in the pipeline makes an
internet / pretrained-weights / external-API call: the model is built from
scratch with plain PyTorch and a self-contained character tokeniser.

CLI override examples:
    python train.py --phase1_epochs 5 --lr 1e-4 --self_play_rounds 3
    python inference.py --device cpu --val_words 500
    python sanity_checks.py --data_dir /path/to/data
"""
import os
import sys


# ---------------------------------------------------------------------------
# CLI argument parsing (runs at import time, safe for notebooks)
# ---------------------------------------------------------------------------
def _parse_cli():
    """Parse ``--key value`` pairs from sys.argv.  Returns dict of overrides.
    Boolean flags use ``--flag`` / ``--no-flag``."""
    _BOOL_KEYS = {"final_full_data", "use_amp", "val_at_end_of_phase1",
                  "val_at_end_of_selfplay"}
    _KNOWN = {
        "data_dir", "base_dir", "checkpoint_dir", "best_model_path",
        "load_checkpoint",
        "seed", "val_split_ratio", "final_full_data",
        "model_dim", "num_heads", "num_layers", "ff_dim", "dropout",
        "max_seq_len", "max_word_len",
        "batch_size", "val_batch_size", "grad_accum_steps",
        "phase1_epochs", "self_play_rounds", "self_play_epochs_per_round",
        "max_train_steps", "max_mc_words_per_epoch",
        "self_play_batch", "mc_mix_ratio", "max_wrong_guesses",
        "correct_guess_prob",
        "lr", "min_lr", "weight_decay", "lr_scheduler", "warmup_steps",
        "warmup_ratio", "max_grad_norm",
        "device", "use_amp", "val_words", "num_workers",
        # Exhaustive state generation
        "exhaustive_cap_per_word", "exhaustive_ratio", "wrong_letters_min",
        "wrong_letters_max", "wrong_letters_prob",
        # Candidate inference
        "candidate_alpha", "candidate_min_size", "candidate_max_size",
        # Misc
        "load_checkpoint_phase", "load_checkpoint_epoch", "load_checkpoint_round",
        "load_checkpoint_global_step",
    }
    # Map CLI key (lowercase) -> module-level global name (UPPERCASE)
    _KEY_MAP = {k: k.upper() for k in _KNOWN}

    overrides = {}
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:].replace("-", "_")
        if key not in _KNOWN:
            i += 1
            continue
        if key in _BOOL_KEYS:
            overrides[key] = True
            if i + 1 < len(argv) and argv[i + 1] == "no":
                overrides[key] = False
                i += 1
        else:
            if i + 1 >= len(argv):
                raise ValueError(f"Flag --{key} requires a value")
            overrides[key] = argv[i + 1]
            i += 1
        i += 1
    return overrides, _KEY_MAP

_CLI_RAW, _KEY_MAP = _parse_cli()


def _coerce(val_str, default):
    """Coerce a string value to match the type of `default`."""
    if isinstance(default, bool):
        return val_str.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(val_str)
    if isinstance(default, float):
        return float(val_str)
    if default is None:
        try:
            return int(val_str)
        except ValueError:
            try:
                return float(val_str)
            except ValueError:
                return val_str
    return val_str


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
METRICS_LOG = os.path.join(BASE_DIR, "training_log.csv")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOAD_CHECKPOINT = ""  # path to checkpoint to resume training from (empty = train from scratch)

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
# ---------------------------------------------------------------------------
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
A_CODE = ord("a")
NUM_LETTERS = 26

PAD = 26
MASK = 27
CLS = 28
SEP = 29
VOCAB_SIZE = 30

# ---------------------------------------------------------------------------
# Model dimensions (CANINE-S style character transformer)
# ---------------------------------------------------------------------------
MODEL_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
FF_DIM = 1024
DROPOUT = 0.1
MAX_SEQ_LEN = 64
MAX_WORD_LEN = 32

# Layout: CLS(1) + guessed(26) + SEP(1) + word(32) = 60
# The model also receives two 26-dim masks appended after the word tokens.
# With correct_mask(26) + wrong_mask(26), total = 60 + 52 = 112 <= 128
# We increase MAX_SEQ_LEN to 128 to accommodate the masks.
MAX_SEQ_LEN = 128
LAYOUT_GUESS_START = 1
LAYOUT_GUESS_END = 27   # exclusive
LAYOUT_SEP = 27
LAYOUT_WORD_START = 28
# Correct-letter mask: positions 60..85 (26 tokens)
LAYOUT_CORRECT_START = 60
# Wrong-letter mask: positions 86..111 (26 tokens)
LAYOUT_WRONG_START = 86
# Total used: 112 tokens
LAYOUT_LEN = 60  # base layout for backward compatibility (CLS + 26 guessed + SEP + 32 word)
LAYOUT_TOTAL_LEN = 112  # full length including correct/wrong masks

# Correct/wrong mask token values: 1 = present, 0 = absent
CORRECT_ONE = 1
CORRECT_ZERO = 0
WRONG_ONE = 1
WRONG_ZERO = 0

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

MAX_TRAIN_STEPS = None
MAX_MC_WORDS_PER_EPOCH = 0

SELF_PLAY_BATCH = 4096
MC_MIX_RATIO = 0.30
MAX_WRONG_GUESSES = 6

# ---------------------------------------------------------------------------
# Bias sampling for MC state generation
# ---------------------------------------------------------------------------
CORRECT_GUESS_PROB = 0.40

# ---------------------------------------------------------------------------
# Exhaustive legal state generation
# ---------------------------------------------------------------------------
EXHAUSTIVE_CAP_PER_WORD = 64   # max exhaustive subsets per word (0 = unlimited)
EXHAUSTIVE_RATIO = 0.70        # fraction of training data from exhaustive states
WRONG_LETTERS_MIN = 0          # min wrong letters to attach to a state
WRONG_LETTERS_MAX = 5          # max wrong letters (max 5 for non-terminal)
WRONG_LETTERS_PROB = 0.50      # probability of adding wrong letters to a state

# ---------------------------------------------------------------------------
# Optimiser / scheduler
# ---------------------------------------------------------------------------
LR = 3e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = "cosine"       # "cosine" or "linear_warmup" or "constant"
WARMUP_STEPS = 500
WARMUP_RATIO = 0.0             # if > 0, warmup_steps = total_steps * WARMUP_RATIO
MAX_GRAD_NORM = 1.0

# ---------------------------------------------------------------------------
# Candidate inference scoring
# ---------------------------------------------------------------------------
CANDIDATE_ALPHA = 0.70         # weight for CANINE probability in combined score
CANDIDATE_MIN_SIZE = 1         # min candidate set size to use candidate score
CANDIDATE_MAX_SIZE = 5000      # max candidate set size to use candidate score

# ---------------------------------------------------------------------------
# Mixed precision / device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda"
USE_AMP = True
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
VAL_WORDS = None
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

# ---------------------------------------------------------------------------
# Checkpoint resume metadata (populated when loading a checkpoint)
# ---------------------------------------------------------------------------
RESUME_PHASE = ""
RESUME_EPOCH = 0
RESUME_ROUND = 0
RESUME_GLOBAL_STEP = 0

# ---------------------------------------------------------------------------
# Apply CLI overrides (after all defaults are defined)
# ---------------------------------------------------------------------------
_mod = sys.modules[__name__]
_applied = []
for _key, _val_str in _CLI_RAW.items():
    _attr = _KEY_MAP[_key]
    _default = getattr(_mod, _attr)
    setattr(_mod, _attr, _coerce(_val_str, _default))
    _applied.append(f"{_attr}={_coerce(_val_str, _default)}")
if _applied:
    print(f"[config] CLI overrides: {', '.join(_applied)}", flush=True)
