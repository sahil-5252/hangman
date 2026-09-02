"""Training pipeline: Phase-1 Monte-Carlo supervised + Phase-2 self-play.

Phase 1: regenerate legal Hangman states from training words each epoch via
biased random sampling (CORRECT_GUESS_PROB), train with the frequency-weighted
soft-target cross-entropy.  This is plain supervised learning (no RL/PPO/DQN).

Phase 2: the model plays the training words itself in batches to produce
self-play trajectories; these are mixed with fresh Monte-Carlo states and
trained with the SAME soft-target CrossEntropyLoss.  The best model is saved
according to held-out validation Hangman win rate.

Key design:
  - Optimizer persists across ALL epochs and rounds (never recreated).
  - Scheduler is created once with total training horizon and saved/restored.
  - Checkpoints save model + optimizer + scheduler + resume metadata.
"""
import os
import csv
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


class MetricsLogger:
    """Append-only CSV logger for training and validation metrics."""

    COLUMNS = [
        "timestamp", "phase", "epoch", "round",
        "train_loss", "train_acc", "lr",
        "val_win_rate", "val_avg_wrong", "val_avg_guesses",
        "val_solved", "val_total",
        "checkpoint_path", "notes",
    ]

    def __init__(self, path=cfg.METRICS_LOG):
        self.path = path
        self._init_file()

    def _init_file(self):
        write_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if write_header:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.COLUMNS)

    def log(self, phase, epoch, train_loss=None, train_acc=None, lr=None,
            val_metrics=None, round_n=None, checkpoint_path="", notes=""):
        from datetime import datetime, timezone
        row = {c: "" for c in self.COLUMNS}
        row["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        row["phase"] = phase
        row["epoch"] = epoch
        row["round"] = round_n if round_n is not None else ""
        row["train_loss"] = f"{train_loss:.6f}" if train_loss is not None else ""
        row["train_acc"] = f"{train_acc:.6f}" if train_acc is not None else ""
        row["lr"] = f"{lr:.8f}" if lr is not None else ""
        if val_metrics:
            row["val_win_rate"] = f"{val_metrics.get('win_rate', ''):.6f}" if 'win_rate' in val_metrics else ""
            row["val_avg_wrong"] = f"{val_metrics.get('avg_wrong', ''):.6f}" if 'avg_wrong' in val_metrics else ""
            row["val_avg_guesses"] = f"{val_metrics.get('avg_total_guesses', ''):.6f}" if 'avg_total_guesses' in val_metrics else ""
            row["val_solved"] = val_metrics.get("solved", "")
            row["val_total"] = val_metrics.get("total", "")
        row["checkpoint_path"] = checkpoint_path
        row["notes"] = notes
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row[c] for c in self.COLUMNS])


def get_device():
    return cfg.DEVICE


def build_model():
    model = CanineHangmanModel()
    return model


def make_optim(model):
    """Create AdamW optimizer.  This is called ONCE and reused across all phases."""
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
    """Create LR scheduler.

    Warmup is done via a LambdaLR wrapper. Cosine annealing is applied
    AFTER warmup, with MIN_LR as the floor.
    """
    warmup = cfg.WARMUP_STEPS if cfg.WARMUP_STEPS > 0 else int(total_steps * cfg.WARMUP_RATIO)

    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        # Cosine annealing from 1.0 down to MIN_LR/LR
        min_ratio = cfg.MIN_LR / cfg.LR
        if total_steps <= warmup:
            return min_ratio
        progress = (step - warmup) / max(1, total_steps - warmup)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def get_total_training_steps():
    """Estimate total training steps across all phases for scheduler."""
    # Phase 1: PHASE1_EPOCHS * (len(mc_words) / BATCH_SIZE)
    # Phase 2: SELF_PLAY_ROUNDS * SELF_PLAY_EPOCHS_PER_ROUND * (len(mc_words) / BATCH_SIZE)
    steps_per_epoch = max(1, cfg.MAX_TRAIN_STEPS or 0)
    if steps_per_epoch == 0:
        # Estimate from data
        mc_len = cfg.MAX_MC_WORDS_PER_EPOCH if cfg.MAX_MC_WORDS_PER_EPOCH and cfg.MAX_MC_WORDS_PER_EPOCH > 0 else 200000
        steps_per_epoch = max(1, mc_len // cfg.BATCH_SIZE)

    total = cfg.PHASE1_EPOCHS * steps_per_epoch
    total += cfg.SELF_PLAY_ROUNDS * cfg.SELF_PLAY_EPOCHS_PER_ROUND * steps_per_epoch
    return max(1, total)


def save_checkpoint(model, optim, scheduler, path, metrics=None,
                    phase="", epoch=0, rnd=0, global_step=0):
    """Save full checkpoint: model + optimizer + scheduler + resume metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "phase": phase,
        "epoch": epoch,
        "round": rnd,
        "global_step": global_step,
    }
    torch.save(state, path)


def load_checkpoint(model, optim, scheduler, path, device):
    """Load full checkpoint: model + optimizer + scheduler + resume metadata.
    Returns (model, optim, scheduler, resume_info_dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optim is not None and "optim_state_dict" in ckpt and ckpt["optim_state_dict"] is not None:
        optim.load_state_dict(ckpt["optim_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    resume_info = {
        "phase": ckpt.get("phase", ""),
        "epoch": ckpt.get("epoch", 0),
        "round": ckpt.get("round", 0),
        "global_step": ckpt.get("global_step", 0),
        "metrics": ckpt.get("metrics"),
    }
    return model, optim, scheduler, resume_info


def train_epoch(model, dataset, optim, scheduler, scaler, loss_fn, epoch, phase,
                max_steps=None, device=cfg.DEVICE, global_step=0):
    """Train for one epoch.  Returns (loss, steps, accuracy, updated_global_step)."""
    model.train()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate, drop_last=False)
    if len(dataset) == 0:
        return 0.0, 0, 0.0, global_step
    use_amp = cfg.USE_AMP and device.type != "cpu"
    step = 0
    running_loss = 0.0
    correct = 0
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

        # Step scheduler every batch (persists across epochs)
        if scheduler is not None:
            scheduler.step()
            # Enforce MIN_LR
            for pg in optim.param_groups:
                if pg["lr"] < cfg.MIN_LR:
                    pg["lr"] = cfg.MIN_LR

        running_loss += loss.item()
        # accuracy: argmax of logits vs argmax of soft targets
        correct += (logits.argmax(dim=-1) == targets.argmax(dim=-1)).sum().item()
        seen += targets.shape[0]
        step += 1
        global_step += 1
        last_lr = optim.param_groups[0]["lr"]
        pbar.set_postfix({"loss": f"{running_loss/max(1,step):.4f}",
                          "acc": f"{correct/max(1,seen):.4f}",
                          "lr": f"{last_lr:.2e}", "steps": step})
        if max_steps is not None and step >= max_steps:
            break
    pbar.close()
    return running_loss / max(1, step), step, correct / max(1, seen), global_step


@torch.no_grad()
def run_validation(agent, val_words, phase, epoch, round_n=None):
    tag = f"{phase} epoch {epoch}"
    if round_n is not None:
        tag += f" round {round_n}"
    desc = f"Validation {tag}"
    metrics = validate(agent, val_words, desc=desc)
    return metrics


def run_pipeline():
    set_seed(cfg.SEED)
    device = torch.device(get_device())
    print(f"[setup] device={device} amp={cfg.USE_AMP} seed={cfg.SEED}", flush=True)

    train_words, val_words = prepare_splits()
    print(f"[data] train={len(train_words)} val={len(val_words)}", flush=True)

    logger = MetricsLogger()
    model = build_model().to(device)

    # Create optimizer ONCE — persists across all phases
    optim = make_optim(model)

    # Create scheduler ONCE with total training horizon
    total_steps = get_total_training_steps()
    scheduler = make_scheduler(optim, total_steps)
    print(f"[setup] total_training_steps={total_steps} warmup={cfg.WARMUP_STEPS} "
          f"min_lr={cfg.MIN_LR}", flush=True)

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.USE_AMP and device.type != "cpu"))
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    best_path = cfg.BEST_MODEL_PATH
    ckpt_dir = cfg.CHECKPOINT_DIR
    global_step = 0

    mc_words = train_words
    if cfg.MAX_MC_WORDS_PER_EPOCH and cfg.MAX_MC_WORDS_PER_EPOCH > 0:
        mc_words = train_words[:cfg.MAX_MC_WORDS_PER_EPOCH]

    # Resume from checkpoint if specified
    start_phase1_epoch = 1
    start_selfplay_round = 1
    start_selfplay_epoch = 0
    if cfg.LOAD_CHECKPOINT and os.path.exists(cfg.LOAD_CHECKPOINT):
        model, optim, scheduler, resume_info = load_checkpoint(
            model, optim, scheduler, cfg.LOAD_CHECKPOINT, device)
        global_step = resume_info["global_step"]
        resume_phase = resume_info["phase"]
        print(f"[setup] loaded checkpoint: {cfg.LOAD_CHECKPOINT}", flush=True)
        print(f"[setup] resume: phase={resume_phase} epoch={resume_info['epoch']} "
              f"round={resume_info['round']} step={global_step}", flush=True)
        if resume_info["metrics"]:
            print(f"[setup] checkpoint metrics: {resume_info['metrics']}", flush=True)

        # Determine where to resume
        if resume_phase.startswith("SelfPlay"):
            start_phase1_epoch = cfg.PHASE1_EPOCHS + 1
            start_selfplay_round = resume_info["round"]
            start_selfplay_epoch = resume_info["epoch"]
        elif resume_phase == "Phase1":
            start_phase1_epoch = resume_info["epoch"] + 1
            start_selfplay_round = 1
            start_selfplay_epoch = 0

    # ---------------- Phase 1 ----------------
    for epoch in range(start_phase1_epoch, cfg.PHASE1_EPOCHS + 1):
        print(f"\n=== Phase 1: Monte-Carlo supervised | epoch {epoch} ===", flush=True)
        dataset = generate_mc_states(mc_words,
                                     np.random.RandomState(cfg.SEED + epoch),
                                     desc=f"MC states e{epoch}")

        loss, steps, train_acc, global_step = train_epoch(
            model, dataset, optim, scheduler, scaler, loss_fn, epoch, "Phase1",
            max_steps=cfg.MAX_TRAIN_STEPS, device=device, global_step=global_step)

        # Save per-epoch checkpoint with full state
        ep_ckpt_path = os.path.join(ckpt_dir, f"phase1_e{epoch}.pt")
        save_checkpoint(model, optim, scheduler, ep_ckpt_path,
                        {"loss": loss, "epoch": epoch},
                        phase="Phase1", epoch=epoch, global_step=global_step)
        print(f"[checkpoint] saved {ep_ckpt_path}", flush=True)

        val_metrics = {}
        if val_words:
            agent = HangmanAgent(model)
            val_metrics = run_validation(agent, val_words, "Phase1", epoch)
            if val_metrics["win_rate"] > best_val:
                best_val = val_metrics["win_rate"]
                save_checkpoint(model, optim, scheduler, best_path, val_metrics,
                                phase="Phase1", epoch=epoch, global_step=global_step)
                print(f"[best] new best val win_rate={best_val:.4f} -> {best_path}", flush=True)
        else:
            print(f"[Phase1] epoch {epoch} loss={loss:.4f}", flush=True)

        last_lr = optim.param_groups[0]["lr"]
        logger.log("Phase1", epoch, train_loss=loss, train_acc=train_acc,
                    lr=last_lr, val_metrics=val_metrics if val_metrics else None,
                    checkpoint_path=ep_ckpt_path)

    # ---------------- Phase 2: self-play ----------------
    if cfg.SELF_PLAY_ROUNDS > 0:
        # Re-load best model if we have validation
        if os.path.exists(best_path):
            model, optim, scheduler, _ = load_checkpoint(
                model, optim, scheduler, best_path, device)
            print(f"[self-play] resumed best model (win_rate={best_val:.4f})", flush=True)

    for rnd in range(start_selfplay_round, cfg.SELF_PLAY_ROUNDS + 1):
        sub_start = start_selfplay_epoch + 1 if rnd == start_selfplay_round else 1
        for sub in range(sub_start, cfg.SELF_PLAY_EPOCHS_PER_ROUND + 1):
            print(f"\n=== Phase 2: self-play | round {rnd} epoch {sub} ===", flush=True)
            agent = HangmanAgent(model)
            input_ids, attn, targets = rollout_selfplay(
                agent, mc_words, desc=f"Self-play r{rnd}e{sub}")
            n_self = input_ids.shape[0]

            # Build mixed dataset
            dataset = _MixedDataset(input_ids, attn, targets, mc_words,
                                    cfg.SEED + 1000 + rnd * 10 + sub)

            loss, steps, train_acc, global_step = train_epoch(
                model, dataset, optim, scheduler, scaler, loss_fn, sub,
                f"SelfPlay r{rnd}",
                max_steps=cfg.MAX_TRAIN_STEPS, device=device, global_step=global_step)

            # Save per-epoch checkpoint with full state
            ep_ckpt_path = os.path.join(ckpt_dir, f"selfplay_r{rnd}_e{sub}.pt")
            save_checkpoint(model, optim, scheduler, ep_ckpt_path,
                            {"loss": loss, "epoch": sub, "round": rnd},
                            phase=f"SelfPlay r{rnd}", epoch=sub, rnd=rnd,
                            global_step=global_step)
            print(f"[checkpoint] saved {ep_ckpt_path}", flush=True)

            val_metrics = {}
            if val_words:
                agent = HangmanAgent(model)
                val_metrics = run_validation(agent, val_words, "SelfPlay", sub, round_n=rnd)
                if val_metrics["win_rate"] > best_val:
                    best_val = val_metrics["win_rate"]
                    save_checkpoint(model, optim, scheduler, best_path, val_metrics,
                                    phase=f"SelfPlay r{rnd}", epoch=sub, rnd=rnd,
                                    global_step=global_step)
                    print(f"[best] new best val win_rate={best_val:.4f} -> {best_path}", flush=True)
            else:
                print(f"[SelfPlay] r{rnd} e{sub} loss={loss:.4f}", flush=True)

            last_lr = optim.param_groups[0]["lr"]
            logger.log("SelfPlay", sub, train_loss=loss, train_acc=train_acc,
                        lr=last_lr, val_metrics=val_metrics if val_metrics else None,
                        round_n=rnd, checkpoint_path=ep_ckpt_path,
                        notes=f"selfplay_states={n_self}")

    # Final save
    final_path = os.path.join(cfg.CHECKPOINT_DIR, "final_model.pt")
    save_checkpoint(model, optim, scheduler, final_path, {"best_val": best_val},
                    phase="done", global_step=global_step)
    print(f"[done] final model -> {final_path}; best_val={best_val:.4f}", flush=True)
    print(f"[done] training log -> {cfg.METRICS_LOG}", flush=True)
    return model, HangmanAgent(model), best_val


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
