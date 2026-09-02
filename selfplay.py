"""Batched self-play rollout.

The current model plays training words itself (greedy, argmax over unmasked
letters) generating *legal* Hangman trajectories.  Each visited state is
labelled with the frequency-weighted soft target derived from the hidden word
(the same target used in Phase-1 Monte-Carlo training).  No repeated guesses
are ever emitted because already-guessed letters are masked to -inf.
"""
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from data import CHAR_TO_ID, soft_target_vector, LAYOUT_LEN
from evaluate import _build_chunk_tokens_extended


@torch.no_grad()
def rollout_selfplay(agent, words, device=None, batch_size=cfg.SELF_PLAY_BATCH,
                     max_wrong=cfg.MAX_WRONG_GUESSES, desc="Self-play"):
    """Rollout model vs. words; collect (input_ids, attn, targets) tensors.

    words : list[str]  (training words only)
    Returns: (input_ids [N,L] int64, attn [N,L] int64, targets [N,26] float32)
    """
    if len(words) == 0:
        return (torch.zeros(0, cfg.LAYOUT_TOTAL_LEN, dtype=torch.long),
                torch.zeros(0, cfg.LAYOUT_TOTAL_LEN, dtype=torch.long),
                torch.zeros(0, cfg.NUM_LETTERS, dtype=torch.float32))

    agent.eval()
    if device is None:
        device = agent.device
    amp_ctx = torch.amp.autocast("cuda", enabled=(cfg.USE_AMP and device.type != "cpu"))

    all_ids, all_attn, all_tgt = [], [], []

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
        done = np.zeros(B, dtype=bool)
        step_states = []  # per-step lists for this chunk

        for step in range(cfg.NUM_LETTERS):
            if done.all():
                break
            active = ~done
            active_idx = np.where(active)[0]
            if len(active_idx) == 0:
                break
            a_words = [chunk[i] for i in active_idx]
            ids, am, gm = _build_chunk_tokens_extended(
                a_words, revealed[active_idx],
                [guessed_char_lists[i] for i in active_idx])
            ids_d = ids.to(device); am_d = am.to(device); gm_d = gm.to(device)
            with amp_ctx:
                logits = agent.model(ids_d, am_d)
            logits = logits.masked_fill(gm_d, float("-inf"))
            nxt = logits.argmax(dim=-1).cpu().numpy()

            # record (state, target) for every active word at this step
            for j, bi in enumerate(active_idx):
                gstr = guessed_char_lists[bi]
                tgt = soft_target_vector(chunk[bi], _guessed_bool(gstr))
                step_states.append((ids[j].numpy(), am[j].numpy(), tgt))

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
                if guess_char in chunk[bi]:
                    revealed[bi] |= (word_ids_arr[bi] == guess_id)
                    wl = min(word_lens[bi], W)
                    if wl > 0 and revealed[bi, :wl].all():
                        done[bi] = True
                else:
                    wrong[bi] += 1
                    if wrong[bi] >= max_wrong:
                        done[bi] = True

        for ids_j, am_j, tgt_j in step_states:
            all_ids.append(ids_j)
            all_attn.append(am_j)
            all_tgt.append(tgt_j)
        pbar.set_postfix({"states": len(all_ids)})

    input_ids = torch.from_numpy(np.stack(all_ids))
    attn = torch.from_numpy(np.stack(all_attn))
    targets = torch.from_numpy(np.stack(all_tgt))
    return input_ids, attn, targets


def _guessed_bool(gstr):
    m = np.zeros(cfg.NUM_LETTERS, dtype=bool)
    for c in gstr:
        m[CHAR_TO_ID[c]] = True
    return m
