"""Training pipeline: Phase-1 Monte-Carlo supervised + Phase-2 self-play.

Phase 1: regenerate legal Hangman states from training words each epoch via
biased random sampling (CORRECT_GUESS_PROB), train with the frequency-weighted
soft-target cross-entropy.  This is plain supervised learning (no RL/PPO/DQN).

Phase 2: the model plays the training words itself in batches to produce
self-play trajectories; these are mixed with fresh Monte-Carlo states and
trained with the SAME soft-target CrossEntropyLoss.  The best model is saved
according to held-out validation Hangman win rate.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config as cfg
from data import set_seed, generate_mc_states, load_splits, prepare_splits, \
    load_train_test, collate
from model import CanineHangmanModel, HangmanAgent
from evaluate import validate
from selfplay import rollout_selfplay


def get_device():
    return cfg.DEVICE


def build_model():
    model = CanineHangmanModel()
    return model


def make_optim(model):
    decay = []; no_decay = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.LR)


def make_scheduler(opt, total_steps):
    if cfg.LR_SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
    elif cfg.LR_SCHEDULER == "linear_warmup":
        def lr_lambda(step):
            if step < cfg.WARMUP_STEPS:
                return step / max(1, cfg.WARMUP_STEPS)
            return max(0.0, (total_steps - step) / max(1, total_steps - cfg.WARMUP_STEPS))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, total_steps), gamma=1.0)


def train_epoch(model, dataset, optim, scheduler, scaler, loss_fn, epoch, phase,
                max_steps=None, device=cfg.DEVICE):
    model.train()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate, drop_last=False)
    if len(dataset) == 0:
        return 0.0, 0
    use_amp = cfg.USE_AMP and device.type != "cpu"
    step = 0
    running_loss = 0.0
    seen = 0
    pbar = tqdm(loader, desc=f"[{phase} epoch {epoch}] train", unit="batch")
    for batch in pbar:
        input_ids, attn, targets = batch
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        targets = targets.to(device)

        optim.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_ids, attn)
        loss = loss_fn(logits, targets)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            optim.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()
        seen += 1
        step += 1
        pbar.set_postfix({"loss": f"{running_loss/max(1,seen):.4f}", "steps": step})
        if max_steps is not None and step >= max_steps:
            break
    pbar.close()
    return running_loss / max(1, seen), step


@torch.no_grad()
def run_validation(agent, val_words, phase, epoch, round_n=None):
    tag = f"{phase} epoch {epoch}"
    if round_n is not None:
        tag += f" round {round_n}"
    desc = f"Validation {tag}"
    metrics = validate(agent, val_words, desc=desc)
    return metrics


def save_checkpoint(model, path, metrics=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, path)


def run_pipeline():
    set_seed(cfg.SEED)
    device = torch.device(get_device())
    # allow numpy bool scalar in soft targets:
    print(f"[setup] device={device} amp={cfg.USE_AMP} seed={cfg.SEED}", flush=True)

    train_words, val_words = prepare_splits()
    print(f"[data] train={len(train_words)} val={len(val_words)}", flush=True)

    model = build_model().to(device)
    agent = HangmanAgent(model)
    optim = make_optim(model)
    # total phase1 steps estimate
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.USE_AMP and device.type != "cpu"))

    best_val = -1.0
    best_path = cfg.BEST_MODEL_PATH

    mc_words = train_words
    if cfg.MAX_MC_WORDS_PER_EPOCH and cfg.MAX_MC_WORDS_PER_EPOCH > 0:
        mc_words = train_words[:cfg.MAX_MC_WORDS_PER_EPOCH]

    # ---------------- Phase 1 ----------------
    for epoch in range(1, cfg.PHASE1_EPOCHS + 1):
        print(f"\n=== Phase 1: Monte-Carlo supervised | epoch {epoch} ===", flush=True)
        dataset = generate_mc_states(mc_words,
                                     np.random.RandomState(cfg.SEED + epoch),
                                     desc=f"MC states e{epoch}")
        optim_cur = make_optim(model)  # fresh optimiser per phase-1 epoch
        sched = make_scheduler(optim_cur, max(1, len(dataset) // cfg.BATCH_SIZE))
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg.USE_AMP and device.type != "cpu"))
        loss, steps = train_epoch(model, dataset, optim_cur, sched, scaler,
                                  loss_fn, epoch, "Phase1",
                                  max_steps=cfg.MAX_TRAIN_STEPS, device=device)
        agent.model = model
        if val_words:
            metrics = run_validation(agent, val_words, "Phase1", epoch)
            if metrics["win_rate"] > best_val:
                best_val = metrics["win_rate"]
                save_checkpoint(model, best_path, metrics)
                print(f"[best] new best val win_rate={best_val:.4f} -> {best_path}", flush=True)
        else:
            print(f"[Phase1] epoch {epoch} loss={loss:.4f}", flush=True)

    # ---------------- Phase 2: self-play ----------------
    if cfg.SELF_PLAY_ROUNDS > 0:
        # re-load best model if we have validation
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            agent = HangmanAgent(model)
            print(f"[self-play] resumed best val model (win_rate={ckpt.get('metrics',{}).get('win_rate',0):.4f})", flush=True)

    for rnd in range(1, cfg.SELF_PLAY_ROUNDS + 1):
        for sub in range(cfg.SELF_PLAY_EPOCHS_PER_ROUND):
            print(f"\n=== Phase 2: self-play | round {rnd} epoch {sub+1} ===", flush=True)
            input_ids, attn, targets = rollout_selfplay(
                agent, mc_words, desc=f"Self-play r{rnd}e{sub+1}")
            n_self = input_ids.shape[0]
            # build a dataset that mixes in fresh MC states to preserve distribution
            dataset = _MixedDataset(input_ids, attn, targets, mc_words,
                                    cfg.SEED + 1000 + rnd * 10 + sub)
            optim_cur = make_optim(model)
            sched = make_scheduler(optim_cur, max(1, len(dataset) // cfg.BATCH_SIZE))
            scaler = torch.amp.GradScaler("cuda", enabled=(cfg.USE_AMP and device.type != "cpu"))
            loss, steps = train_epoch(model, dataset, optim_cur, sched, scaler,
                                      loss_fn, sub + 1, f"SelfPlay r{rnd}",
                                      max_steps=cfg.MAX_TRAIN_STEPS, device=device)
            agent.model = model
            if val_words:
                metrics = run_validation(agent, val_words, "SelfPlay", sub + 1, round_n=rnd)
                if metrics["win_rate"] > best_val:
                    best_val = metrics["win_rate"]
                    save_checkpoint(model, best_path, metrics)
                    print(f"[best] new best val win_rate={best_val:.4f} -> {best_path}", flush=True)
            else:
                print(f"[SelfPlay] r{rnd} e{sub+1} loss={loss:.4f}", flush=True)

    # final save
    final_path = os.path.join(cfg.CHECKPOINT_DIR, "final_model.pt")
    save_checkpoint(model, final_path, {"best_val": best_val})
    print(f"[done] final model -> {final_path}; best_val={best_val:.4f}", flush=True)
    return model, agent, best_val


class _MixedDataset(torch.utils.data.Dataset):
    """Mix a block of self-play states with fresh Monte-Carlo states to keep
    the original training distribution alive (cfg.MC_MIX_RATIO fraction)."""

    def __init__(self, sp_ids, sp_attn, sp_tgt, mc_words, seed):
        self.sp_ids = sp_ids
        self.sp_attn = sp_attn
        self.sp_tgt = sp_tgt
        mc_ds = generate_mc_states(mc_words[: max(1, len(mc_words) // 2)],
                                   np.random.RandomState(seed + 7),
                                   desc="MC mix")
        # sample a subset of MC equal to MC_MIX_RATIO of self-play count
        n_mc = max(1, int(len(sp_ids) * cfg.MC_MIX_RATIO))
        idx = np.random.RandomState(seed + 3).permutation(len(mc_ds))[:n_mc]
        self.mc_ids = torch.from_numpy(mc_ds.input_ids[idx])
        self.mc_attn = torch.from_numpy(mc_ds.attn_mask[idx])
        self.mc_tgt = torch.from_numpy(mc_ds.targets[idx])

    def __len__(self):
        return len(self.sp_ids)

    def __getitem__(self, idx):
        if idx < len(self.sp_ids):
            return (self.sp_ids[idx], self.sp_attn[idx], self.sp_tgt[idx])
        m = idx - len(self.sp_ids)
        return (self.mc_ids[m], self.mc_attn[m], self.mc_tgt[m])


if __name__ == "__main__":
    run_pipeline()


