"""Batched six-strike Hangman validation simulator.

The simulator is fully off-policy during validation: it does NOT train.  It
runs many words in parallel by padding each chunk to the chunk's max word
length and stepping the whole chunk one guess at a time.
"""
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from data import CHAR_TO_ID, LAYOUT_LEN
from model import HangmanAgent


@torch.no_grad()
def _chunk_step(agent, input_ids, attn_mask, guessed_mask, device, amp_ctx):
    """Single batched forward -> argmax guess, respecting guessed_mask."""
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    guessed_mask = guessed_mask.to(device)
    with amp_ctx:
        logits = agent.model(input_ids, attn_mask)
    logits = logits.masked_fill(guessed_mask, float("-inf"))
    nxt = logits.argmax(dim=-1).cpu()
    return nxt  # [B]


def _build_chunk_tokens(chunk_words, revealed, guessed_char_lists, max_len):
    """Build token tensors for a chunk at the start of each step.

    chunk_words: list[str]
    revealed: np.ndarray [B, W] bool
    guessed_char_lists: list[str] of guessed letters per word
    """
    B = len(chunk_words)
    input_ids = np.full((B, max_len), cfg.PAD, dtype=np.int64)
    attn = np.ones((B, max_len), dtype=np.int64)
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
            if pos < max_len:
                input_ids[b, pos] = CHAR_TO_ID[c] if rev[i] else cfg.MASK
    return (torch.from_numpy(input_ids), torch.from_numpy(attn),
            torch.from_numpy(guessed_mask))


@torch.no_grad()
def validate(agent, val_words, device=None, batch_size=cfg.VAL_BATCH_SIZE,
             max_wrong=cfg.MAX_WRONG_GUESSES, desc="Validation"):
    """Run the full six-strike hangman simulator on val_words.

    Returns dict: win_rate, avg_wrong, avg_total_guesses, solved, total.
    Does NOT train the model.
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
            ids, am, gm = _build_chunk_tokens(
                a_words, a_revealed, a_guessed, LAYOUT_LEN)
            nxt = _chunk_step(agent, ids, am, gm, device, amp_ctx)
            nxt = nxt.numpy()

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
