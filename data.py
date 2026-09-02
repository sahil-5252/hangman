"""Data utilities for the Meltwater Hangman pipeline.

Responsibilities
----------------
* Load ``train.txt`` / ``test.txt`` (a-z words only).
* Produce a reproducible 90/10 word-level train/val split and persist CSVs.
* Generate *legal* Hangman states from a Monte-Carlo biased random sampler.
* Compute the frequency-weighted soft target distribution.
* A self-contained character tokeniser (no external tokenizer / weights).
"""
import csv
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config as cfg

# ---------------------------------------------------------------------------
# Alphabet helpers
# ---------------------------------------------------------------------------
ALPHABET = cfg.ALPHABET
CHAR_TO_ID = {ch: i for i, ch in enumerate(ALPHABET)}
ID_TO_CHAR = {i: ch for ch, i in CHAR_TO_ID.items()}
assert len(ALPHABET) == cfg.NUM_LETTERS == 26


def word_to_ids(word: str):
    """Map a pure a-z word to character ids 0..25."""
    return [CHAR_TO_ID[c] for c in word]


def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------
def load_words(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(path)}", unit="lines"):
            w = line.strip().lower()
            if not w:
                continue
            assert all(c in CHAR_TO_ID for c in w), f"Non a-z char in {w}"
            words.append(w)
    return words


def load_train_test():
    train_words = load_words(cfg.TRAIN_FILE)
    test_words = load_words(cfg.TEST_FILE)
    return train_words, test_words


# ---------------------------------------------------------------------------
# Reproducible 90/10 split
# ---------------------------------------------------------------------------
def make_split(train_words):
    """Deterministic 90/10 word split.  Returns (train_words, val_words)."""
    rng = np.random.RandomState(cfg.SEED)
    idx = rng.permutation(len(train_words))
    n_val = int(len(train_words) * cfg.VAL_SPLIT_RATIO)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    val_words = [train_words[i] for i in val_idx]
    train_words_split = [train_words[i] for i in train_idx]
    return train_words_split, val_words


def write_csv(words, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for word in words:
            w.writerow([word])


def prepare_splits():
    train_words, _ = load_train_test()
    if cfg.FINAL_FULL_DATA:
        train_words_split = train_words
        val_words = train_words[:0]
    else:
        train_words_split, val_words = make_split(train_words)
    write_csv(train_words_split, cfg.TRAIN_SPLIT_CSV)
    write_csv(val_words, cfg.VAL_SPLIT_CSV)
    return train_words_split, val_words


def load_splits():
    def _read(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    train_words = _read(cfg.TRAIN_SPLIT_CSV)
    val_words = _read(cfg.VAL_SPLIT_CSV)
    return train_words, val_words


# ---------------------------------------------------------------------------
# Soft target (frequency weighted)
# ---------------------------------------------------------------------------
def soft_target_vector(word, guessed_mask):
    """target(letter) = remaining hidden occurrences / total remaining hidden.

    guessed_mask: bool numpy array length 26, True if letter already guessed.
    Letters not in the word and already guessed -> zero probability.
    """
    counts = np.zeros(cfg.NUM_LETTERS, dtype=np.float64)
    total = 0
    for c in word:
        ci = CHAR_TO_ID[c]
        if guessed_mask[ci]:
            continue  # already guessed -> zero
        counts[ci] += 1
        total += 1
    out = np.zeros(cfg.NUM_LETTERS, dtype=np.float64)
    if total > 0:
        out = counts / total
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Monte-Carlo biased random state generation
# ---------------------------------------------------------------------------
def simulate_game_mc(word, rng, correct_rate=cfg.CORRECT_GUESS_PROB,
                     max_wrong=cfg.MAX_WRONG_GUESSES):
    """Generate (state, target) pairs via legal biased-random hangman play.

    Each guess reveals ALL occurrences of that letter.  Already-guessed
    letters are never repeated.
    """
    word_ids = np.array(word_to_ids(word), dtype=np.int64)
    n = len(word_ids)
    revealed = np.zeros(n, dtype=bool)  # True = shown
    guessed_mask = np.zeros(cfg.NUM_LETTERS, dtype=bool)
    guessed_chars = []
    num_wrong = 0
    states = []
    targets = []

    while (not revealed.all()) and num_wrong < max_wrong:
        # soft target for the current state
        tgt = soft_target_vector(word, guessed_mask)
        states.append(("".join(guessed_chars), tuple(revealed.tolist()), word))
        targets.append(tgt)

        # biased guess
        unguessed = [c for c in ALPHABET if not guessed_mask[CHAR_TO_ID[c]]]
        if not unguessed:
            break
        # correct-letter pool
        correct_pool = [c for c in unguessed if c in word]
        if correct_pool and rng.random() < correct_rate:
            guess = rng.choice(correct_pool)
        else:
            guess = rng.choice(unguessed)

        ci = CHAR_TO_ID[guess]
        guessed_mask[ci] = True
        guessed_chars.append(guess)
        if guess in word:
            revealed[word_ids == ci] = True
        else:
            num_wrong += 1

    solved = bool(revealed.all())
    return states, np.array(targets), solved


# ---------------------------------------------------------------------------
# Tokeniser (fixed layout for fast batched simulation)
#   [CLS] guessed[0..25] [SEP] word[0..MAX_WORD_LEN-1]
# ---------------------------------------------------------------------------
LAYOUT_LEN = 1 + cfg.NUM_LETTERS + 1 + cfg.MAX_WORD_LEN  # 60
assert LAYOUT_LEN <= cfg.MAX_SEQ_LEN, "layout exceeds MAX_SEQ_LEN"


def encode_state(guessed_chars, revealed_flags, word, max_len=LAYOUT_LEN):
    """Return (input_ids, attn_mask) for a single hangman state.

    guessed_chars: string of unique guessed letters in guess order.
    revealed_flags: list/bool-array length len(word): True = shown.
    word: the (hidden) target word (only used by the simulator, never by the
          model as a raw embedding of the answer).
    """
    ids = [cfg.PAD] * max_len
    attn = [1] * max_len
    ids = [cfg.CLS] + [cfg.PAD] * (max_len - 1)
    for i, c in enumerate(guessed_chars[:cfg.NUM_LETTERS]):
        ids[cfg.LAYOUT_GUESS_START + i] = CHAR_TO_ID[c]
    ids[cfg.LAYOUT_SEP] = cfg.SEP
    for i, c in enumerate(word[:cfg.MAX_WORD_LEN]):
        pos = cfg.LAYOUT_WORD_START + i
        if pos < max_len:
            ids[pos] = CHAR_TO_ID[c] if revealed_flags[i] else cfg.MASK
    return ids, attn


def encode_batch(guessed_char_lists, revealed_arr, words, max_len=LAYOUT_LEN):
    """Vectorised batch encoder.

    guessed_char_lists: list[str] (one per sample)
    revealed_arr: np.ndarray [B, W] bool  (W = len of each word, <= MAX_WORD_LEN)
    words: list[str]
    Returns input_ids [B, L] int64, attn [B, L] int64, guessed_mask [B, 26] bool
    """
    B = len(words)
    input_ids = np.full((B, max_len), cfg.PAD, dtype=np.int64)
    guessed_mask = np.zeros((B, cfg.NUM_LETTERS), dtype=bool)

    input_ids[:, 0] = cfg.CLS
    input_ids[:, cfg.LAYOUT_SEP] = cfg.SEP

    for b, gstr in enumerate(guessed_char_lists):
        for i, c in enumerate(gstr[:cfg.NUM_LETTERS]):
            input_ids[b, cfg.LAYOUT_GUESS_START + i] = CHAR_TO_ID[c]
            guessed_mask[b, CHAR_TO_ID[c]] = True

    for b, (word, revealed) in enumerate(zip(words, revealed_arr)):
        for i, c in enumerate(word[:cfg.MAX_WORD_LEN]):
            pos = cfg.LAYOUT_WORD_START + i
            input_ids[b, pos] = CHAR_TO_ID[c] if revealed[i] else cfg.MASK

    attn = np.ones((B, max_len), dtype=np.int64)
    return input_ids, attn, guessed_mask


# ---------------------------------------------------------------------------
# Dataset over pre-encoded state tensors
# ---------------------------------------------------------------------------
class HangmanStateDataset(Dataset):
    def __init__(self, input_ids, attn_mask, targets):
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.targets = targets

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attn_mask[idx], self.targets[idx])


def collate(batch):
    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    input_ids = torch.tensor(np.stack([_t(b[0]) for b in batch]), dtype=torch.long)
    attn = torch.tensor(np.stack([_t(b[1]) for b in batch]), dtype=torch.long)
    targets = torch.tensor(np.stack([_t(b[2]) for b in batch]), dtype=torch.float32)
    return input_ids, attn, targets


def make_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=cfg.NUM_WORKERS, collate_fn=collate,
                      pin_memory=True if cfg.DEVICE != "cpu" else False,
                      drop_last=False)


# ---------------------------------------------------------------------------
# Generate a pool of Monte-Carlo states from a word list
# ---------------------------------------------------------------------------
def generate_mc_states(words, rng=None, correct_rate=cfg.CORRECT_GUESS_PROB,
                       max_wrong=cfg.MAX_WRONG_GUESSES, desc="MC states"):
    if rng is None:
        rng = np.random.RandomState(cfg.SEED)
    all_state_tuples = []
    all_targets = []
    total = 0
    total_solved = 0
    for w in tqdm(words, desc=desc, unit="words"):
        states, targets, solved = simulate_game_mc(w, rng, correct_rate, max_wrong)
        all_state_tuples.extend(states)
        all_targets.extend(targets)
        total += 1
        total_solved += int(solved)
    print(f"[{desc}] states={len(all_state_tuples)} words={total} "
          f"solved={total_solved}/{total} win_rate={total_solved/max(1,total):.3f}", flush=True)
    if not all_state_tuples:
        # avoid empty encoder crash
        empty = (("", (False,), "a"))
        all_state_tuples = [empty]
        all_targets = [soft_target_vector("a", np.zeros(26, bool))]
    n = len(all_state_tuples)
    input_ids = np.full((n, LAYOUT_LEN), cfg.PAD, dtype=np.int64)
    attn = np.ones((n, LAYOUT_LEN), dtype=np.int64)
    for i, (gstr, revealed, word) in enumerate(all_state_tuples):
        ids, am = encode_state(gstr, revealed, word)
        input_ids[i] = ids
        attn[i] = am
    targets = np.array(all_targets, dtype=np.float32)
    return HangmanStateDataset(input_ids, attn, targets)
