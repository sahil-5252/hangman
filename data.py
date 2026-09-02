"""Data utilities for the Meltwater Hangman pipeline.

Responsibilities
----------------
* Load ``train.txt`` / ``test.txt`` (a-z words only).
* Produce a reproducible 90/10 word-level train/val split and persist CSVs.
* Generate *legal* Hangman states via:
  - Exhaustive legal reveal states (letter-subset enumeration).
  - Monte-Carlo biased random sampler.
  - Randomized wrong-letter histories for each state.
* Compute the frequency-weighted soft target distribution.
* A self-contained character tokeniser (no external tokenizer / weights).
* Candidate index builder for inference.
"""
import csv
import os
import itertools
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
# Tokeniser (fixed layout for fast batched simulation)
#   [CLS] guessed[0..25] [SEP] word[0..MAX_WORD_LEN-1]
#   Layout is LAYOUT_LEN=60 for backward compat with existing code.
#   The extended layout with correct/wrong masks is handled separately.
# ---------------------------------------------------------------------------
LAYOUT_LEN = 60  # 1 + 26 + 1 + 32


def encode_state(guessed_chars, revealed_flags, word, max_len=LAYOUT_LEN):
    """Return (input_ids, attn_mask) for a single hangman state.

    guessed_chars: string of unique guessed letters in guess order.
    revealed_flags: list/bool-array length len(word): True = shown.
    word: the (hidden) target word (only used by the simulator, never by the
          model as a raw embedding of the answer).
    """
    ids = [cfg.CLS] + [cfg.PAD] * (max_len - 1)
    attn = [1] * max_len
    for i, c in enumerate(guessed_chars[:cfg.NUM_LETTERS]):
        ids[cfg.LAYOUT_GUESS_START + i] = CHAR_TO_ID[c]
    ids[cfg.LAYOUT_SEP] = cfg.SEP
    for i, c in enumerate(word[:cfg.MAX_WORD_LEN]):
        pos = cfg.LAYOUT_WORD_START + i
        if pos < max_len:
            ids[pos] = CHAR_TO_ID[c] if revealed_flags[i] else cfg.MASK
    return ids, attn


def encode_state_extended(guessed_chars, revealed_flags, word,
                          correct_mask, wrong_mask):
    """Return (input_ids, attn_mask) for a hangman state with correct/wrong masks.

    Layout:
        [CLS] guessed[0..25] [SEP] word[0..31] correct_mask[0..25] wrong_mask[0..25]
    Total: 1 + 26 + 1 + 32 + 26 + 26 = 112 tokens

    correct_mask[i] = 1 if letter i has been guessed correctly (is in the word)
    wrong_mask[i] = 1 if letter i has been guessed incorrectly (not in the word)
    """
    ids = [cfg.CLS] + [cfg.PAD] * (cfg.LAYOUT_TOTAL_LEN - 1)
    attn = [1] * cfg.LAYOUT_TOTAL_LEN

    # Guessed letters in order
    for i, c in enumerate(guessed_chars[:cfg.NUM_LETTERS]):
        ids[cfg.LAYOUT_GUESS_START + i] = CHAR_TO_ID[c]

    # Separator
    ids[cfg.LAYOUT_SEP] = cfg.SEP

    # Word with masks
    for i, c in enumerate(word[:cfg.MAX_WORD_LEN]):
        pos = cfg.LAYOUT_WORD_START + i
        if pos < cfg.LAYOUT_TOTAL_LEN:
            ids[pos] = CHAR_TO_ID[c] if revealed_flags[i] else cfg.MASK

    # Correct-letter mask (26 dims)
    for i in range(26):
        ids[cfg.LAYOUT_CORRECT_START + i] = cfg.CORRECT_ONE if correct_mask[i] else cfg.CORRECT_ZERO

    # Wrong-letter mask (26 dims)
    for i in range(26):
        ids[cfg.LAYOUT_WRONG_START + i] = cfg.WRONG_ONE if wrong_mask[i] else cfg.WRONG_ZERO

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


def encode_batch_extended(guessed_char_lists, revealed_arr, words,
                          correct_masks, wrong_masks):
    """Vectorised batch encoder with correct/wrong masks.

    guessed_char_lists: list[str]
    revealed_arr: np.ndarray [B, W] bool
    words: list[str]
    correct_masks: np.ndarray [B, 26] bool
    wrong_masks: np.ndarray [B, 26] bool
    Returns input_ids [B, 112] int64, attn [B, 112] int64, guessed_mask [B, 26] bool
    """
    B = len(words)
    L = cfg.LAYOUT_TOTAL_LEN
    input_ids = np.full((B, L), cfg.PAD, dtype=np.int64)
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
            if pos < L:
                input_ids[b, pos] = CHAR_TO_ID[c] if revealed[i] else cfg.MASK

    # Correct-letter mask
    for b in range(B):
        for i in range(26):
            input_ids[b, cfg.LAYOUT_CORRECT_START + i] = (
                cfg.CORRECT_ONE if correct_masks[b, i] else cfg.CORRECT_ZERO)

    # Wrong-letter mask
    for b in range(B):
        for i in range(26):
            input_ids[b, cfg.LAYOUT_WRONG_START + i] = (
                cfg.WRONG_ONE if wrong_masks[b, i] else cfg.WRONG_ZERO)

    attn = np.ones((B, L), dtype=np.int64)
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
# Exhaustive legal state generation (letter-subset based)
# ---------------------------------------------------------------------------
def _enumerate_legal_subsets(word, cap=cfg.EXHAUSTIVE_CAP_PER_WORD, rng=None):
    """Enumerate legal reveal subsets for a word.

    A legal state reveals ALL occurrences of each selected letter.
    We enumerate subsets of unique letters in the word.
    If the number of subsets exceeds `cap`, we sample randomly.

    Yields: (revealed_mask_array,) where revealed_mask is bool array of len(word)
    """
    unique_letters = list(set(word))
    n_unique = len(unique_letters)

    # Total subsets = 2^n_unique (excluding empty set and full set)
    # We skip empty (nothing revealed) and full (word solved, no training signal)
    total_subsets = (1 << n_unique) - 2  # minus empty and full
    if total_subsets <= 0:
        # Word has 1 unique letter -> only 1 meaningful partial state (some revealed, some not)
        # For very small alphabets, just yield a single state with some revealed
        if n_unique == 1:
            revealed = np.zeros(len(word), dtype=bool)
            # reveal all but one occurrence
            letter = unique_letters[0]
            count = word.count(letter)
            revealed_count = max(1, count - 1)
            c = 0
            for i, ch in enumerate(word):
                if ch == letter and c < revealed_count:
                    revealed[i] = True
                    c += 1
            yield revealed
        return

    if cap <= 0 or total_subsets <= cap:
        # Enumerate all subsets
        for r in range(1, n_unique):  # skip 0 (empty) and n_unique (solved)
            for combo in itertools.combinations(range(n_unique), r):
                revealed = np.zeros(len(word), dtype=bool)
                for idx in combo:
                    letter = unique_letters[idx]
                    for i, c in enumerate(word):
                        if c == letter:
                            revealed[i] = True
                yield revealed
    else:
        # Sample `cap` random subsets
        if rng is None:
            rng = np.random.RandomState(cfg.SEED)
        sampled = set()
        attempts = 0
        while len(sampled) < cap and attempts < cap * 10:
            # Random subset: each unique letter included with prob 0.5
            mask_bits = rng.random(n_unique) < 0.5
            if not mask_bits.any() or mask_bits.all():
                attempts += 1
                continue
            key = tuple(mask_bits.tolist())
            if key in sampled:
                attempts += 1
                continue
            sampled.add(key)
            revealed = np.zeros(len(word), dtype=bool)
            for idx in range(n_unique):
                if mask_bits[idx]:
                    letter = unique_letters[idx]
                    for i, c in enumerate(word):
                        if c == letter:
                            revealed[i] = True
            yield revealed
            attempts += 1


def _random_wrong_letters(word, revealed_mask, guessed_mask, rng,
                          n_wrong=None):
    """Generate a random set of wrong letters for a given state.

    Wrong letters must:
    - Not overlap with revealed/correct letters
    - Not already be guessed
    - Be unique
    - At most 5 for a non-terminal state

    Returns: list of wrong letter characters
    """
    # Letters that are IN the word but not yet revealed
    word_letters = set(word)
    revealed_letters = set()
    for i, c in enumerate(word):
        if revealed_mask[i]:
            revealed_letters.add(c)

    # Letters that are wrong (in alphabet, not in word, not yet guessed)
    wrong_pool = [c for c in ALPHABET
                  if c not in word_letters
                  and not guessed_mask[CHAR_TO_ID[c]]]

    if not wrong_pool:
        return []

    if n_wrong is None:
        # Random number of wrong letters
        max_possible = min(cfg.WRONG_LETTERS_MAX, len(wrong_pool))
        if max_possible <= 0:
            return []
        n_wrong = rng.randint(cfg.WRONG_LETTERS_MIN, max_possible + 1)
    else:
        n_wrong = min(n_wrong, len(wrong_pool), cfg.WRONG_LETTERS_MAX)

    if n_wrong <= 0:
        return []

    chosen = rng.choice(wrong_pool, size=n_wrong, replace=False).tolist()
    return chosen


def generate_exhaustive_states(word, rng=None, cap=cfg.EXHAUSTIVE_CAP_PER_WORD):
    """Generate exhaustive legal states for a single word.

    For each legal reveal subset, also attach random wrong-letter histories.
    Returns: list of (guessed_chars_str, revealed_tuple, word_str), targets
    """
    if rng is None:
        rng = np.random.RandomState(cfg.SEED)

    states = []
    targets = []

    for revealed_mask in _enumerate_legal_subsets(word, cap=cap, rng=rng):
        # Compute which letters are correctly guessed (revealed)
        revealed_letters = set()
        for i, c in enumerate(word):
            if revealed_mask[i]:
                revealed_letters.add(c)

        # Build guessed_chars: the revealed letters (in arbitrary but consistent order)
        guessed_chars = sorted(revealed_letters)

        # Build guessed_mask for wrong-letter generation
        guessed_mask = np.zeros(cfg.NUM_LETTERS, dtype=bool)
        for c in guessed_chars:
            guessed_mask[CHAR_TO_ID[c]] = True

        # Generate wrong letters
        wrong_chars = _random_wrong_letters(word, revealed_mask, guessed_mask, rng)

        # Full guessed set = correct letters + wrong letters
        all_guessed = guessed_chars + wrong_chars
        all_guessed_str = "".join(all_guessed)

        # Compute target: soft distribution over hidden letters
        tgt = soft_target_vector(word, guessed_mask)

        states.append((all_guessed_str, tuple(revealed_mask.tolist()), word))
        targets.append(tgt)

    return states, np.array(targets, dtype=np.float32) if targets else np.zeros((0, 26), dtype=np.float32)


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
# Generate a pool of states (exhaustive + MC hybrid)
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


def generate_exhaustive_states_dataset(words, rng=None, desc="Exhaustive states"):
    """Generate exhaustive legal states for all words, with random wrong-letter histories.

    Returns a HangmanStateDataset with extended layout (112 tokens).
    """
    if rng is None:
        rng = np.random.RandomState(cfg.SEED)

    all_input_ids = []
    all_attn = []
    all_targets = []
    total_states = 0

    for w in tqdm(words, desc=desc, unit="words"):
        states, targets = generate_exhaustive_states(w, rng=rng,
                                                     cap=cfg.EXHAUSTIVE_CAP_PER_WORD)
        if len(states) == 0:
            continue

        # Encode each state with extended layout
        for (gstr, revealed, word), tgt in zip(states, targets):
            # Compute correct/wrong masks
            guessed_mask = np.zeros(cfg.NUM_LETTERS, dtype=bool)
            for c in gstr:
                guessed_mask[CHAR_TO_ID[c]] = True

            word_letters = set(word)
            correct_mask = np.zeros(cfg.NUM_LETTERS, dtype=bool)
            wrong_mask = np.zeros(cfg.NUM_LETTERS, dtype=bool)
            for c in gstr:
                ci = CHAR_TO_ID[c]
                if c in word_letters:
                    correct_mask[ci] = True
                else:
                    wrong_mask[ci] = True

            ids, am = encode_state_extended(gstr, revealed, word,
                                            correct_mask, wrong_mask)
            all_input_ids.append(ids)
            all_attn.append(am)
            all_targets.append(tgt)
            total_states += 1

    print(f"[{desc}] total_states={total_states} words={len(words)}", flush=True)

    if total_states == 0:
        # Fallback: single dummy state
        dummy_ids = [cfg.CLS] + [cfg.PAD] * (cfg.LAYOUT_TOTAL_LEN - 1)
        dummy_attn = [1] * cfg.LAYOUT_TOTAL_LEN
        dummy_tgt = soft_target_vector("a", np.zeros(26, bool))
        all_input_ids = [dummy_ids]
        all_attn = [dummy_attn]
        all_targets = [dummy_tgt]

    input_ids = np.array(all_input_ids, dtype=np.int64)
    attn = np.array(all_attn, dtype=np.int64)
    targets = np.array(all_targets, dtype=np.float32)
    return HangmanStateDataset(input_ids, attn, targets)


def generate_hybrid_dataset(mc_words, rng=None, desc="Hybrid states"):
    """Generate a hybrid dataset: exhaustive states + MC states.

    The proportion is controlled by cfg.EXHAUSTIVE_RATIO.
    Returns a HangmanStateDataset with extended layout (112 tokens).
    """
    if rng is None:
        rng = np.random.RandomState(cfg.SEED)

    # Generate exhaustive states
    exhaustive_ds = generate_exhaustive_states_dataset(
        mc_words, rng=np.random.RandomState(rng.randint(0, 2**31)),
        desc=f"{desc} exhaustive")

    # Generate MC states (with extended encoding)
    mc_states_list = []
    mc_targets_list = []
    for w in tqdm(mc_words, desc=f"{desc} MC", unit="words"):
        states, targets, _ = simulate_game_mc(w, rng, cfg.CORRECT_GUESS_PROB,
                                               cfg.MAX_WRONG_GUESSES)
        for (gstr, revealed, word), tgt in zip(states, targets):
            guessed_mask = np.zeros(cfg.NUM_LETTERS, dtype=bool)
            for c in gstr:
                guessed_mask[CHAR_TO_ID[c]] = True

            word_letters = set(word)
            correct_mask = np.zeros(cfg.NUM_LETTERS, dtype=bool)
            wrong_mask = np.zeros(cfg.NUM_LETTERS, dtype=bool)
            for c in gstr:
                ci = CHAR_TO_ID[c]
                if c in word_letters:
                    correct_mask[ci] = True
                else:
                    wrong_mask[ci] = True

            ids, am = encode_state_extended(gstr, revealed, word,
                                            correct_mask, wrong_mask)
            mc_states_list.append(ids)
            mc_attn_list.append(am) if False else None
            mc_targets_list.append(tgt)

    # Combine exhaustive + MC
    all_ids = [exhaustive_ds.input_ids]
    all_attn = [exhaustive_ds.attn_mask]
    all_tgt = [exhaustive_ds.targets]

    if mc_states_list:
        all_ids.append(np.array(mc_states_list, dtype=np.int64))
        all_attn.append(np.array([1] * len(mc_states_list), dtype=np.int64))  # placeholder
        all_tgt.append(np.array(mc_targets_list, dtype=np.float32))

    # Blend based on EXHAUSTIVE_RATIO
    n_exh = len(exhaustive_ds)
    n_mc = len(mc_states_list)
    total = n_exh + n_mc

    if total == 0:
        return exhaustive_ds

    # For simplicity, concatenate all states
    combined_ids = np.concatenate([exhaustive_ds.input_ids,
                                    np.array(mc_states_list, dtype=np.int64) if mc_states_list else np.zeros((0, cfg.LAYOUT_TOTAL_LEN), dtype=np.int64)])
    combined_attn = np.concatenate([exhaustive_ds.attn_mask,
                                     np.ones((n_mc, cfg.LAYOUT_TOTAL_LEN), dtype=np.int64) if mc_states_list else np.zeros((0, cfg.LAYOUT_TOTAL_LEN), dtype=np.int64)])
    combined_tgt = np.concatenate([exhaustive_ds.targets,
                                    np.array(mc_targets_list, dtype=np.float32) if mc_targets_list else np.zeros((0, 26), dtype=np.float32)])

    print(f"[{desc}] combined: exhaustive={n_exh} mc={n_mc} total={total}", flush=True)
    return HangmanStateDataset(combined_ids, combined_attn, combined_tgt)


# ---------------------------------------------------------------------------
# Candidate index for inference
# ---------------------------------------------------------------------------
class CandidateIndex:
    """Efficient retrieval of candidate training words for a given Hangman state.

    Builds inverted indexes by word length, positional characters, and known
    wrong letters for fast filtering.
    """

    def __init__(self, train_words):
        self.train_words = train_words
        self._build_index()

    def _build_index(self):
        """Build inverted indexes for fast candidate retrieval."""
        # Word length index
        self.length_index = {}  # length -> list of (word, word_id)
        for i, w in enumerate(self.train_words):
            lw = len(w)
            if lw not in self.length_index:
                self.length_index[lw] = []
            self.length_index[lw].append((w, i))

        # Positional character index: (length, position, char) -> set of word_ids
        self.positional_index = {}
        for i, w in enumerate(self.train_words):
            lw = len(w)
            for pos, c in enumerate(w):
                key = (lw, pos, c)
                if key not in self.positional_index:
                    self.positional_index[key] = set()
                self.positional_index[key].add(i)

        # Letter presence index: (length, letter) -> set of word_ids
        self.letter_index = {}
        for i, w in enumerate(self.train_words):
            lw = len(w)
            for c in set(w):
                key = (lw, c)
                if key not in self.letter_index:
                    self.letter_index[key] = set()
                self.letter_index[key].add(i)

        print(f"[CandidateIndex] built: {len(self.train_words)} words, "
              f"{len(self.length_index)} lengths", flush=True)

    def retrieve(self, word_len, revealed_flags, word_pattern,
                 wrong_letters, max_size=cfg.CANDIDATE_MAX_SIZE):
        """Retrieve candidate words matching the current Hangman state.

        word_len: length of the hidden word
        revealed_flags: bool array, True where letter is revealed
        word_pattern: the actual word (used for revealed positions)
        wrong_letters: set of letters known to be wrong

        Returns: list of candidate words
        """
        # Start with words of correct length
        if word_len not in self.length_index:
            return []

        candidates = set(wid for _, wid in self.length_index[word_len])

        # Filter by revealed positions
        for pos in range(word_len):
            if revealed_flags[pos] and pos < len(word_pattern):
                char = word_pattern[pos]
                key = (word_len, pos, char)
                if key in self.positional_index:
                    candidates &= self.positional_index[key]
                else:
                    return []  # no words match this constraint

        # Filter by wrong letters (candidates must NOT contain wrong letters)
        for c in wrong_letters:
            key = (word_len, c)
            if key in self.letter_index:
                candidates -= self.letter_index[key]

        # Convert to word list
        result = []
        word_list = self.length_index.get(word_len, [])
        for w, wid in word_list:
            if wid in candidates:
                result.append(w)
                if len(result) >= max_size:
                    break

        return result

    def letter_probability(self, candidates, guessed_mask):
        """Compute letter probability distribution over the candidate set.

        P_C(l) = (# candidate words containing l) / (# candidate words)
        Only considers unguessed letters.
        """
        probs = np.zeros(cfg.NUM_LETTERS, dtype=np.float64)
        if not candidates:
            return probs

        for w in candidates:
            seen = set()
            for c in w:
                ci = CHAR_TO_ID[c]
                if not guessed_mask[ci] and ci not in seen:
                    probs[ci] += 1
                    seen.add(ci)

        total = len(candidates)
        if total > 0:
            probs /= total

        return probs
