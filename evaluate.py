"""Batched six-strike Hangman validation simulator.

The simulator is fully off-policy during validation: it does NOT train.  It
runs many words in parallel by padding each chunk to the chunk's max word
length and stepping the whole chunk one guess at a time.
"""
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from data import CHAR_TO_ID, LAYOUT_LEN, CandidateIndex
from model import HangmanAgent


def _build_chunk_tokens_extended(chunk_words, revealed, guessed_char_lists):
    """Build token tensors for a chunk with correct/wrong masks.

    Returns input_ids [B, 112], attn_mask [B, 112], guessed_mask [B, 26]
    """
    B = len(chunk_words)
    L = cfg.LAYOUT_TOTAL_LEN
    input_ids = np.full((B, L), cfg.PAD, dtype=np.int64)
    attn = np.ones((B, L), dtype=np.int64)
    guessed_mask = np.zeros((B, cfg.NUM_LETTERS), dtype=bool)

    input_ids[:, 0] = cfg.CLS
    input_ids[:, cfg.LAYOUT_SEP] = cfg.SEP

    for b, gstr in enumerate(guessed_char_lists):
        for i, c in enumerate(gstr[:cfg.NUM_LETTERS]):
            input_ids[b, cfg.LAYOUT_GUESS_START + i] = CHAR_TO_ID[c]
            guessed_mask[b, CHAR_TO_ID[c]] = True

    for b, (word, rev) in enumerate(zip(chunk_words, revealed)):
        for i, c in enumerate(word[:cfg.MAX_WORD_LEN]):
            pos = cfg.LAYOUT_WORD_START + i
            if pos < L:
                input_ids[b, pos] = CHAR_TO_ID[c] if rev[i] else cfg.MASK

    # Correct-letter mask
    for b, gstr in enumerate(guessed_char_lists):
        for c in gstr:
            ci = CHAR_TO_ID[c]
            if c in chunk_words[b]:
                input_ids[b, cfg.LAYOUT_CORRECT_START + ci] = cfg.CORRECT_ONE
            else:
                input_ids[b, cfg.LAYOUT_WRONG_START + ci] = cfg.WRONG_ONE

    return (torch.from_numpy(input_ids), torch.from_numpy(attn),
            torch.from_numpy(guessed_mask))


def _combined_score(agent, cand_idx, chunk_words, revealed, guessed_char_lists,
                    word_ids_arr, device, amp_ctx):
    """Compute combined CANINE+candidate+six-strike score for next guess.

    Returns next_letter_id [B] for each word in the chunk.
    """
    B = len(chunk_words)
    input_ids, attn, gm = _build_chunk_tokens_extended(
        chunk_words, revealed, guessed_char_lists)

    # CANINE probabilities
    canine_probs = agent.predict_proba(input_ids.to(device), attn.to(device),
                                        gm.to(device)).cpu().numpy()  # [B, 26]

    # Candidate probabilities (if index available)
    candidate_probs = np.zeros((B, cfg.NUM_LETTERS), dtype=np.float64)
    if cand_idx is not None:
        for b, word in enumerate(chunk_words):
            word_len = len(word)
            revealed_flags = revealed[b]
            wrong_letters = set()
            for c in guessed_char_lists[b]:
                if c not in word:
                    wrong_letters.add(c)
            candidates = cand_idx.retrieve(word_len, revealed_flags, word,
                                           wrong_letters)
            if candidates:
                cprobs = cand_idx.letter_probability(
                    candidates, guessed_mask=gm[b].numpy())
                candidate_probs[b] = cprobs

    # Combined score: alpha * CANINE + (1-alpha) * candidate
    alpha = cfg.CANDIDATE_ALPHA
    combined = alpha * canine_probs + (1.0 - alpha) * candidate_probs

    # Six-strike risk penalty
    # Penalise guesses that are likely to be wrong (reduce risk of hitting 6)
    # For now, just use the combined score directly
    next_id = combined.argmax(axis=-1)  # [B]
    return next_id


@torch.no_grad()
def validate(agent, val_words, device=None, batch_size=cfg.VAL_BATCH_SIZE,
             max_wrong=cfg.MAX_WRONG_GUESSES, desc="Validation",
             cand_index=None):
    """Run the full six-strike hangman simulator on val_words.

    Returns dict: win_rate, avg_wrong, avg_total_guesses, solved, total.
    Does NOT train the model.

    cand_index: optional CandidateIndex for combined scoring.
    """
    if val_words is None or len(val_words) == 0:
        print("[Validation] no words", flush=True)
        return {"win_rate": 0.0, "avg_wrong": 0.0, "avg_total_guesses": 0.0,
                "solved": 0, "total": 0}

    cap = cfg.VAL_WORDS if cfg.VAL_WORDS is not None else len(val_words)
    words = val_words[:cap]
    agent.eval()
    if device is None:
        device = agent.device
    amp_ctx = torch.amp.autocast("cuda", enabled=(cfg.USE_AMP and device.type != "cpu"))

    total_solved = 0
    total_wrong_sum = 0
    total_guesses_sum = 0
    total = len(words)

    pbar = tqdm(range(0, len(words), batch_size), desc=desc, unit="batch")
    for start in pbar:
        chunk = words[start:start + batch_size]
        B = len(chunk)
        wlen = max(len(w) for w in chunk)
        W = min(wlen, cfg.MAX_WORD_LEN)

        word_ids_list = [[CHAR_TO_ID[c] for c in w] for w in chunk]
        word_lens = np.array([len(w) for w in chunk], dtype=np.int64)
        word_ids_arr = np.full((B, W), -1, dtype=np.int64)
        for b, wids in enumerate(word_ids_list):
            wl = min(len(wids), W)
            word_ids_arr[b, :wl] = wids[:wl]
        revealed = np.zeros((B, W), dtype=bool)
        guessed_char_lists = ["" for _ in range(B)]
        wrong = np.zeros(B, dtype=np.int64)
        total_guesses = np.zeros(B, dtype=np.int64)
        solved = np.zeros(B, dtype=bool)
        done = np.zeros(B, dtype=bool)

        for step in range(cfg.NUM_LETTERS):  # max 26 guesses
            if done.all():
                break
            active = ~done
            if not active.any():
                break
            active_idx = np.where(active)[0]
            a_words = [chunk[i] for i in active_idx]
            a_revealed = revealed[active_idx]
            a_guessed = [guessed_char_lists[i] for i in active_idx]

            if cand_index is not None:
                nxt = _combined_score(agent, cand_index, a_words, a_revealed,
                                      a_guessed, word_ids_arr[active_idx],
                                      device, amp_ctx)
            else:
                # Fallback to CANINE-only scoring
                ids, am, gm = _build_chunk_tokens_extended(
                    a_words, a_revealed, a_guessed)
                nxt = agent.predict(ids.to(device), am.to(device),
                                    gm.to(device)).cpu().numpy()

            for j, bi in enumerate(active_idx):
                guess_id = int(nxt[j])
                gstr = guessed_char_lists[bi]
                if chr(ord("a") + guess_id) in gstr:
                    for cid in range(cfg.NUM_LETTERS):
                        if chr(ord("a") + cid) not in gstr:
                            guess_id = cid
                            break
                guess_char = chr(ord("a") + guess_id)
                guessed_char_lists[bi] = gstr + guess_char
                total_guesses[bi] += 1
                if guess_char in chunk[bi]:
                    revealed[bi] |= (word_ids_arr[bi] == guess_id)
                    wl = min(word_lens[bi], W)
                    if wl > 0 and revealed[bi, :wl].all():
                        solved[bi] = True
                        done[bi] = True
                else:
                    wrong[bi] += 1
                    if wrong[bi] >= max_wrong:
                        done[bi] = True

        total_solved += int(solved.sum())
        total_wrong_sum += int(wrong.sum())
        total_guesses_sum += int(total_guesses.sum())
        pbar.set_postfix({
            "win": f"{total_solved / max(1, start + B):.3f}",
            "solved": f"{total_solved}/{total}",
            "avg_wrong": f"{total_wrong_sum / max(1, start + B):.3f}",
        })

    win_rate = total_solved / max(1, total)
    avg_wrong = total_wrong_sum / max(1, total)
    avg_total = total_guesses_sum / max(1, total)
    print(f"[Validation] win_rate={win_rate*100:.2f}% solved={total_solved}/{total} "
          f"avg_wrong={avg_wrong:.3f} avg_total_guesses={avg_total:.3f}", flush=True)
    return {"win_rate": win_rate, "avg_wrong": avg_wrong,
            "avg_total_guesses": avg_total, "solved": int(total_solved),
            "total": total}
