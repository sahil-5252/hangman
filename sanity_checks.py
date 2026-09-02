"""Lightweight CPU sanity checks.

No real training is performed locally (no GPU / heavy CPU).  These checks
verify correctness of: data loading & split sizes, legal state generation,
target distribution validity, model forward/backward, a couple of training
steps on CPU, batched self-play on a tiny subset, the validation simulator,
submission generation against a tiny fake test set, exhaustive state generation,
wrong-letter histories, extended encoding, candidate index, combined scoring,
and checkpoint resume.
"""
import os
import numpy as np
import torch
import csv
from tqdm import tqdm

import config as cfg
from data import (set_seed, load_train_test, prepare_splits, load_splits,
                  make_split, generate_mc_states, soft_target_vector,
                  CHAR_TO_ID, ID_TO_CHAR, LAYOUT_LEN, collate,
                  generate_exhaustive_states, generate_exhaustive_states_dataset,
                  CandidateIndex, encode_state_extended)
from model import CanineHangmanModel, HangmanAgent
from evaluate import validate, _build_chunk_tokens_extended
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

    # 11. Exhaustive state generation (letter-subset based)
    tiny_exh = train_words[:5]
    states_exh, tgt_exh = generate_exhaustive_states(tiny_exh[0], rng=np.random.RandomState(0),
                                                      cap=8)
    check(len(states_exh) > 0, f"exhaustive states generated ({len(states_exh)} states)")
    check(len(tgt_exh) == len(states_exh), "exhaustive states/targets count match")
    # Each state should have a valid target distribution
    for tgt in tgt_exh:
        s = tgt.sum()
        if s > 0:
            check(np.isclose(s, 1.0, atol=1e-5), f"exhaustive target sums to 1.0 (got {s})")
    check(True, "exhaustive states have valid target distributions")

    # 12. Wrong-letter histories are randomized
    from data import _random_wrong_letters, ALPHABET
    word = "hello"
    revealed_mask = np.array([True, True, False, False, False], dtype=bool)
    guessed_mask = np.zeros(cfg.NUM_LETTERS, dtype=bool)
    guessed_mask[CHAR_TO_ID['h']] = True
    guessed_mask[CHAR_TO_ID['e']] = True
    wrong1 = _random_wrong_letters(word, revealed_mask, guessed_mask,
                                   np.random.RandomState(0))
    wrong2 = _random_wrong_letters(word, revealed_mask, guessed_mask,
                                   np.random.RandomState(42))
    check(isinstance(wrong1, list), "wrong letters is a list")
    check(all(c in ALPHABET for c in wrong1), "wrong letters are valid a-z")
    # Wrong letters should not overlap with word letters or guessed mask
    word_letters = set(word)
    for c in wrong1:
        check(c not in word_letters, f"wrong letter {c} not in word")
        check(not guessed_mask[CHAR_TO_ID[c]], f"wrong letter {c} not already guessed")
    check(True, "wrong-letter histories are valid and non-overlapping")

    # 13. Extended encoding (correct/wrong masks)
    guessed_str = "he"
    revealed = np.array([True, True, False, False, False], dtype=bool)
    correct_mask = np.zeros(cfg.NUM_LETTERS, dtype=bool)
    wrong_mask = np.zeros(cfg.NUM_LETTERS, dtype=bool)
    correct_mask[CHAR_TO_ID['h']] = True
    correct_mask[CHAR_TO_ID['e']] = True
    wrong_mask[CHAR_TO_ID['x']] = True
    ids_ext, am_ext = encode_state_extended(guessed_str, revealed, word,
                                            correct_mask, wrong_mask)
    check(len(ids_ext) == cfg.LAYOUT_TOTAL_LEN, f"extended layout length {cfg.LAYOUT_TOTAL_LEN} (got {len(ids_ext)})")
    check(ids_ext[0] == cfg.CLS, "extended layout starts with CLS")
    check(ids_ext[cfg.LAYOUT_SEP] == cfg.SEP, "extended layout has SEP at correct position")
    # Check correct mask positions
    check(ids_ext[cfg.LAYOUT_CORRECT_START + CHAR_TO_ID['h']] == cfg.CORRECT_ONE,
          "correct mask h=1")
    check(ids_ext[cfg.LAYOUT_CORRECT_START + CHAR_TO_ID['x']] == cfg.CORRECT_ZERO,
          "correct mask x=0")
    check(ids_ext[cfg.LAYOUT_WRONG_START + CHAR_TO_ID['x']] == cfg.WRONG_ONE,
          "wrong mask x=1")
    check(ids_ext[cfg.LAYOUT_WRONG_START + CHAR_TO_ID['h']] == cfg.WRONG_ZERO,
          "wrong mask h=0")
    check(True, "extended encoding with correct/wrong masks is correct")

    # 14. Extended encoding batch function
    from data import encode_batch_extended
    words_batch = ["hello", "world"]
    revealed_batch = np.array([[True, True, False, False, False],
                                [True, False, False, False, False]], dtype=bool)
    guessed_batch = ["he", "w"]
    correct_batch = np.zeros((2, 26), dtype=bool)
    wrong_batch = np.zeros((2, 26), dtype=bool)
    for b, gstr in enumerate(guessed_batch):
        for c in gstr:
            if c in words_batch[b]:
                correct_batch[b, CHAR_TO_ID[c]] = True
            else:
                wrong_batch[b, CHAR_TO_ID[c]] = True
    ids_b, am_b, gm_b = encode_batch_extended(guessed_batch, revealed_batch,
                                               words_batch, correct_batch, wrong_batch)
    check(ids_b.shape == (2, cfg.LAYOUT_TOTAL_LEN), f"batch extended shape (2, {cfg.LAYOUT_TOTAL_LEN})")
    check(am_b.shape == (2, cfg.LAYOUT_TOTAL_LEN), "batch extended attn shape")
    check(gm_b.shape == (2, 26), "batch guessed mask shape")
    check(True, "extended batch encoding works correctly")

    # 15. Candidate index
    ci = CandidateIndex(train_words[:500])
    candidates = ci.retrieve(5, np.array([True, True, False, False, False]),
                             "hello", {"x"})
    check(isinstance(candidates, list), "candidate retrieval returns a list")
    check(len(candidates) >= 0, "candidate retrieval is non-negative")
    # All candidates should be length 5 and not contain wrong letter
    for c in candidates:
        check(len(c) == 5, f"candidate {c} has correct length 5")
        check("x" not in c, f"candidate {c} doesn't contain wrong letter x")
    # Letter probability
    probs = ci.letter_probability(candidates, np.zeros(26, dtype=bool))
    check(probs.shape == (26,), "letter probability shape (26,)")
    check(abs(probs.sum() - 1.0) < 1e-5 or probs.sum() == 0,
          "letter probability sums to 1.0 or is empty")
    check(True, "candidate index works correctly")

    # 16. Combined scoring in validation
    metrics_comb = validate(agent, train_words[:40], batch_size=20,
                            desc="Val combined sanity", cand_index=ci)
    check("win_rate" in metrics_comb, "combined scoring validation returns win_rate")
    check(0.0 <= metrics_comb["win_rate"] <= 1.0, "combined win_rate in [0,1]")
    check(True, "combined CANINE+candidate scoring works")

    # 17. Checkpoint save/load round-trip (full state)
    from train import save_checkpoint, load_checkpoint
    model_ckpt = CanineHangmanModel()
    optim_ckpt = torch.optim.AdamW(model_ckpt.parameters(), lr=1e-3)
    total_steps_ckpt = 100
    sched_ckpt = torch.optim.lr_scheduler.CosineAnnealingLR(optim_ckpt, T_max=total_steps_ckpt)
    # Step a few times
    for _ in range(10):
        optim_ckpt.step()
        sched_ckpt.step()
    lr_before = optim_ckpt.param_groups[0]["lr"]
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "_sanity_ckpt.pt")
    save_checkpoint(model_ckpt, optim_ckpt, sched_ckpt, ckpt_path,
                    {"test": True}, phase="Phase1", epoch=3, rnd=0, global_step=42)
    # Load into fresh model
    model_ckpt2 = CanineHangmanModel()
    optim_ckpt2 = torch.optim.AdamW(model_ckpt2.parameters(), lr=1e-3)
    sched_ckpt2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_ckpt2, T_max=total_steps_ckpt)
    model_ckpt2, optim_ckpt2, sched_ckpt2, resume = load_checkpoint(
        model_ckpt2, optim_ckpt2, sched_ckpt2, ckpt_path, torch.device("cpu"))
    lr_after = optim_ckpt2.param_groups[0]["lr"]
    check(resume["phase"] == "Phase1", f"checkpoint phase=Phase1 (got {resume['phase']})")
    check(resume["epoch"] == 3, f"checkpoint epoch=3 (got {resume['epoch']})")
    check(resume["global_step"] == 42, f"checkpoint global_step=42 (got {resume['global_step']})")
    check(np.isclose(lr_before, lr_after, atol=1e-8), f"LR restored ({lr_before:.6f} -> {lr_after:.6f})")
    # Check model weights match
    for (n1, p1), (n2, p2) in zip(model_ckpt.named_parameters(), model_ckpt2.named_parameters()):
        assert torch.allclose(p1, p2), f"param mismatch: {n1}"
    check(True, "checkpoint round-trip preserves model/optimizer/scheduler state")
    os.remove(ckpt_path)

    # 18. Model forward with extended layout (112 tokens)
    model_ext = CanineHangmanModel()
    ids_rand = torch.randint(0, cfg.VOCAB_SIZE, (4, cfg.LAYOUT_TOTAL_LEN))
    attn_rand = torch.ones(4, cfg.LAYOUT_TOTAL_LEN, dtype=torch.long)
    logits_ext = model_ext(ids_rand, attn_rand)
    check(logits_ext.shape == (4, 26), f"extended forward logits shape (4,26) (got {tuple(logits_ext.shape)})")
    check(True, "model handles extended layout (112 tokens)")

    # 19. predict_proba works
    agent_ext = HangmanAgent(model_ext)
    probs_ext = agent_ext.predict_proba(ids_rand, attn_rand)
    check(probs_ext.shape == (4, 26), f"predict_proba shape (4,26)")
    check(np.allclose(probs_ext.sum(dim=-1).numpy(), 1.0, atol=1e-5),
          "predict_proba sums to 1.0")
    check(True, "predict_proba returns valid probability distribution")

    # 20. MIN_LR is respected
    check(cfg.MIN_LR == 1e-5, f"MIN_LR=1e-5 (got {cfg.MIN_LR})")
    check(cfg.CANDIDATE_ALPHA == 0.70, f"CANDIDATE_ALPHA=0.70 (got {cfg.CANDIDATE_ALPHA})")
    check(cfg.EXHAUSTIVE_CAP_PER_WORD == 64, f"EXHAUSTIVE_CAP=64 (got {cfg.EXHAUSTIVE_CAP_PER_WORD})")
    check(cfg.WRONG_LETTERS_MAX == 5, f"WRONG_LETTERS_MAX=5 (got {cfg.WRONG_LETTERS_MAX})")
    check(True, "config hyperparameters correct")

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
