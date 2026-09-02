"""Batched submission generation (Kaggle submission logic).

Reads data/test.txt in original order and, for each word, runs an actual local
six-strike Hangman simulation driven by the trained model.  The model only
ever sees the current board / guessed-letter state.  The hidden test word is
used solely by the local simulator to (a) reveal all occurrences of a guessed
letter and (b) decide the game is solved.

Output: submission.csv  with columns  word_id,guessed_letters_string  (0..249999).
"""
import csv
import os
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from data import load_words, CHAR_TO_ID, LAYOUT_LEN
from evaluate import _build_chunk_tokens
from model import CanineHangmanModel, HangmanAgent


@torch.no_grad()
def generate_submission(agent, test_words, out_path=cfg.BEST_MODEL_PATH,
                        device=None, batch_size=cfg.VAL_BATCH_SIZE,
                        max_wrong=cfg.MAX_WRONG_GUESSES,
                        out_csv="submission.csv"):
    """Simulate all test words and write submission.csv in original order."""
    agent.eval()
    if device is None:
        device = agent.device
    amp_ctx = torch.amp.autocast("cuda", enabled=(cfg.USE_AMP and device.type != "cpu"))
    total = len(test_words)
    rows = []

    pbar = tqdm(range(0, total, batch_size), total=(total + batch_size - 1) // batch_size,
                desc="Submission generation", unit="batch")
    for start in pbar:
        chunk = test_words[start:start + batch_size]
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
        done = np.zeros(B, dtype=bool)

        for step in range(cfg.NUM_LETTERS):
            if done.all():
                break
            active = ~done
            active_idx = np.where(active)[0]
            if len(active_idx) == 0:
                break
            a_words = [chunk[i] for i in active_idx]
            ids, am, gm = _build_chunk_tokens(
                a_words, revealed[active_idx],
                [guessed_char_lists[i] for i in active_idx], LAYOUT_LEN)
            ids_d = ids.to(device)
            am_d = am.to(device)
            gm_d = gm.to(device)
            with amp_ctx:
                logits = agent.model(ids_d, am_d)
            logits = logits.masked_fill(gm_d, float("-inf"))
            nxt = logits.argmax(dim=-1).cpu().numpy()

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
                        done[bi] = True
                else:
                    wrong[bi] += 1
                    if wrong[bi] >= max_wrong:
                        done[bi] = True

        for bi in range(B):
            rows.append((start + bi, guessed_char_lists[bi]))

        pbar.set_postfix({
            "done": f"{start + B}/{total}",
            "avg_wrong": f"{wrong.mean():.2f}",
        })

    pbar.close()

    # Write exactly: word_id,guessed_letters_string, in original test order.
    rows.sort(key=lambda r: r[0])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word_id", "guessed_letters_string"])
        for wid, gstr in rows:
            # hard guarantees required by the schema
            assert isinstance(gstr, str)
            assert all("a" <= c <= "z" for c in gstr), f"bad guess string {gstr!r}"
            assert len(gstr) == len(set(gstr)), f"repeated guess in {gstr!r}"
            w.writerow([wid, gstr])
    print(f"[submission] wrote {len(rows)} rows -> {out_csv}")
    return out_csv


def verify_submission(path, expected_n=None):
    """Schema sanity check on a submission file."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["word_id", "guessed_letters_string"], header
        prev = -1
        for row in reader:
            assert len(row) == 2, row
            wid = int(row[0])
            assert wid == prev + 1, f"word_id mismatch {wid} after {prev}"
            gstr = row[1]
            assert all("a" <= c <= "z" for c in gstr), gstr
            assert len(gstr) == len(set(gstr)), f"repeated guess {gstr!r}"
            prev = wid
            n += 1
    print(f"[verify] OK: {n} rows, contiguous word_id 0..{n-1}, all lowercase a-z, no repeats.")
    if expected_n is not None:
        assert n == expected_n, f"expected {expected_n} rows got {n}"
    return n


if __name__ == "__main__":
    test_words = load_words(cfg.TEST_FILE)
    print(f"[inference] {len(test_words)} test words")
    model = CanineHangmanModel().to(cfg.DEVICE)
    agent = HangmanAgent(model)
    if os.path.exists(cfg.BEST_MODEL_PATH):
        ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[inference] no checkpoint found; using random model.")
    generate_submission(agent, test_words, out_csv="submission.csv")
    verify_submission("submission.csv", expected_n=len(test_words))
