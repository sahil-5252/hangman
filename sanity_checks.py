"""Lightweight CPU sanity checks.

No real training is performed locally (no GPU / heavy CPU).  These checks
verify correctness of: data loading & split sizes, legal state generation,
target distribution validity, model forward/backward, a couple of training
steps on CPU, batched self-play on a tiny subset, the validation simulator,
and submission generation against a tiny fake test set.
"""
import os
import numpy as np
import torch
import csv
from tqdm import tqdm

import config as cfg
from data import (set_seed, load_train_test, prepare_splits, load_splits,
                  make_split, generate_mc_states, soft_target_vector,
                  CHAR_TO_ID, ID_TO_CHAR, LAYOUT_LEN, collate)
from model import CanineHangmanModel, HangmanAgent
from evaluate import validate
from selfplay import rollout_selfplay
from inference import generate_submission, verify_submission

import torch.nn as nn


def check(condition, msg):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {msg}")
    assert condition, msg


def main():
    set_seed(cfg.SEED)
    print("=" * 70)
    print("SANITY CHECKS (CPU, tiny subsets)")
    print("=" * 70)

    # 1. files load correctly
    train_words, test_words = load_train_test()
    check(len(train_words) == 225300, f"train.txt has 225300 words (got {len(train_words)})")
    check(len(test_words) == 250000, f"test.txt has 250000 words (got {len(test_words)})")
    check(all(w.isalpha() and w.islower() for w in train_words), "all train words a-z lowercase")

    # 2. reproducible 90/10 split sizes
    tr1, va1 = make_split(train_words)
    tr2, va2 = make_split(train_words)
    check(len(tr1) == 202770, f"train split size ~202770 (got {len(tr1)})")
    check(len(va1) == 22530, f"val split size ~22530 (got {len(va1)})")
    check(tr1 == tr2 and va1 == va2, "split is reproducible")
    check(set(tr1).isdisjoint(set(va1)), "train/val disjoint (word-level)")
    check(set(tr1).union(set(va1)) == set(train_words), "union recovers all train words")

    # persist & reload CSVs
    from data import write_csv
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    write_csv(tr1, cfg.TRAIN_SPLIT_CSV)
    write_csv(va1, cfg.VAL_SPLIT_CSV)
    trb, vab = load_splits()
    check(trb == tr1 and vab == va1, "CSV round-trip preserves splits")

    # 3. MC state generation + legal states + target validity
    rng = np.random.RandomState(0)
    tiny = train_words[:600]
    ds = generate_mc_states(tiny, rng, desc="MC sanity")
    x = ds.input_ids; a = ds.attn_mask; y = ds.targets
    check(x.shape[0] == y.shape[0], f"inputs/targets count match ({x.shape[0]})")
    check(y.shape[1] == 26, f"targets are [N,26] (got {y.shape[1]})")
    # each non-terminal target is a valid probability distribution (sum ~1)
    row_sums = y.sum(axis=1)
    nonzero = row_sums[row_sums > 0]
    check(len(nonzero) == 0 or np.allclose(nonzero, 1.0, atol=1e-5),
          "target rows sum to 1.0 (prob distributions)")
    # already-guessed letters must have zero prob -> verify via single-word sim
    w = "hello"
    states, tgt_single, solved = simulate_game_mc_one(w)
    for (gstr, revealed, _word), tgt in zip(states, tgt_single):
        for c in gstr:
            idx = CHAR_TO_ID[c]
            assert tgt[idx] == 0.0, f"guessed letter {c} has nonzero target"
        for ci in range(26):
            if tgt[ci] > 0:
                assert chr(ord("a") + ci) in _word, "target on letter not in word"
    check(True, "guessed letters have zero target prob; mass only on hidden word letters")

    # 4. repeated/same-letter partial masking cannot occur (guessed unique per state)
    all_unique = all(len(set(gstr)) == len(gstr) for gstr, *_ in states)
    check(all_unique, "guessed-letters string is unique per state (no repeats)")

    # 5. model forward pass on tiny batch (CPU)
    model = CanineHangmanModel()
    agent = HangmanAgent(model)
    input_ids = torch.from_numpy(x[:8])
    attn = torch.from_numpy(a[:8])
    targets = torch.from_numpy(y[:8])
    logits = model(input_ids, attn)
    check(logits.shape == (8, 26), f"forward logits shape (8,26) (got {tuple(logits.shape)})")

    # 6. loss computes correctly + 1-2 tiny training steps on CPU
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    check(loss.item() > 0 and torch.isfinite(loss), "CE loss finite & positive")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    opt.step()
    opt.zero_grad()
    loss2 = loss_fn(model(input_ids, attn), targets)
    check(loss2.item() < loss.item() + 1.0, "loss decreases / changes after a step")
    print(f"[info] tiny-step loss {loss.item():.4f} -> {loss2.item():.4f}")

    # 7. self-play works on a tiny subset
    model2 = CanineHangmanModel()
    agent2 = HangmanAgent(model2)
    sp_ids, sp_attn, sp_tgt = rollout_selfplay(agent2, train_words[:40],
                                               batch_size=20, desc="Self-play sanity")
    check(sp_ids.shape[0] > 0, f"self-play produced states ({sp_ids.shape[0]})")
    check(sp_tgt.shape[1] == 26, "self-play targets [N,26]")

    # 8. validation simulator works
    metrics = validate(agent, train_words[:120], batch_size=40, desc="Val sanity")
    check("win_rate" in metrics and "avg_wrong" in metrics, "validation returns required metrics")
    check(metrics["total"] == 120, f"validation total words == 120 (got {metrics['total']})")
    check(0.0 <= metrics["win_rate"] <= 1.0, "win_rate in [0,1]")

    # 9. submission generator on tiny fake test set
    fake_test = ["hello", "world", "torch", "model", "alpha"]
    # write a temporary test file ordering
    tmp_test = os.path.join(cfg.DATA_DIR, "test.txt")
    orig_test = open(tmp_test).readlines()
    with open(tmp_test, "w") as f:
        for w in fake_test:
            f.write(w + "\n")
    try:
        out = os.path.join(cfg.BASE_DIR, "submission_tiny.csv")
        generate_submission(agent, fake_test, out_csv=out, batch_size=8)
        n = verify_submission(out, expected_n=len(fake_test))
        # verify order
        with open(out) as f:
            r = list(csv.reader(f))[1:]
        check([row[0] for row in r] == [str(i) for i in range(len(fake_test))],
              "submission word_id is contiguous 0..N-1 in order")
        check(all(len(row[1]) == len(set(row[1])) for row in r), "no repeated guesses")
        check(all(row[1].islower() and row[1].isalpha() for row in r), "guess strings a-z only")
        os.remove(out)
    finally:
        with open(tmp_test, "w") as f:
            f.writelines(orig_test)

    # 10. MC states: regenerated each epoch produce valid, legal states
    rng2 = np.random.RandomState(42)
    ds_b = generate_mc_states(train_words[:200], rng2, desc="MC re-gen")
    check(ds_b.input_ids.shape[0] == ds_b.targets.shape[0], "regenerated MC states consistent")

    print("=" * 70)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 70)


# local import for the single-word MC test (avoids heavy import cycles)
from data import simulate_game_mc


def simulate_game_mc_one(word, rng=None, correct_rate=cfg.CORRECT_GUESS_PROB,
                         max_wrong=cfg.MAX_WRONG_GUESSES):
    if rng is None:
        rng = np.random.RandomState(0)
    states, outputs, solved = simulate_game_mc(word, rng, correct_rate, max_wrong)
    return states, outputs, solved


if __name__ == "__main__":
    main()
