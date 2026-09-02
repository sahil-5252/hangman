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
        "seed", "val_split_ratio", "final_full_data",
        "model_dim", "num_heads", "num_layers", "ff_dim", "dropout",
        "max_seq_len", "max_word_len",
        "batch_size", "val_batch_size", "grad_accum_steps",
        "phase1_epochs", "self_play_rounds", "self_play_epochs_per_round",
        "max_train_steps", "max_mc_words_per_epoch",
        "self_play_batch", "mc_mix_ratio", "max_wrong_guesses",
        "correct_guess_prob",
        "lr", "weight_decay", "lr_scheduler", "warmup_steps", "max_grad_norm",
        "device", "use_amp", "val_words", "num_workers",
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

LAYOUT_GUESS_START = 1
LAYOUT_GUESS_END = 27
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

MAX_TRAIN_STEPS = None
MAX_MC_WORDS_PER_EPOCH = 0

SELF_PLAY_BATCH = 4096
MC_MIX_RATIO = 0.30
MAX_WRONG_GUESSES = 6

CORRECT_GUESS_PROB = 0.40

# ---------------------------------------------------------------------------
# Optimiser / scheduler
# ---------------------------------------------------------------------------
LR = 3e-4
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = "cosine"
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0

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
