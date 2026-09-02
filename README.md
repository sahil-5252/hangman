# Meltwater Hangman — CANINE-S Style Character Transformer

A self-contained pipeline for the Meltwater Hangman hackathon: trains a CANINE-S style character-level Transformer from scratch with Monte-Carlo supervised learning (Phase 1) followed by self-play fine-tuning (Phase 2), and generates a Kaggle submission CSV.

## Architecture

| Component | Details |
|-----------|---------|
| **Model** | CANINE-S style character Transformer Encoder |
| **Params** | 3.19M |
| **Hidden dim** | 256 |
| **Layers / Heads** | 4 / 4 |
| **Feed-forward dim** | 1024 |
| **Input layout** | `[CLS] guessed[0..25] [SEP] word[0..31]` → 60 tokens ≤ 64 max |
| **Output** | 26 logits via CLS pooling (one per a-z letter) |
| **Dropout** | 0.1 |

## Training

### Phase 1 — Monte-Carlo Supervised (3 epochs)
- Legal Hangman states generated via biased random sampler (40% correct-letter bias)
- Frequency-weighted soft targets: `P(letter) = remaining hidden occurrences / total hidden`
- Cosine LR schedule from 3e-4, AdamW with weight decay 1e-4
- 512 batch size, optional AMP on GPU

### Phase 2 — Self-Play Fine-Tuning (2 rounds × 1 epoch)
- Model plays training words itself (greedy argmax, no repeated guesses)
- Each visited state labelled with the same soft-target distribution
- 30% fresh MC states mixed into each self-play batch
- Best model saved by held-out validation win rate

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters and paths |
| `data.py` | Data loading, MC simulation, encoding, dataset |
| `model.py` | CANINE-S Transformer + HangmanAgent wrapper |
| `evaluate.py` | Batched 6-strike validation simulator |
| `selfplay.py` | Batched self-play rollout |
| `train.py` | Phase 1 + Phase 2 training pipeline |
| `inference.py` | Submission generation + schema validation |
| `sanity_checks.py` | CPU sanity checks (all must pass) |
| `notebook.ipynb` | Self-contained Kaggle notebook (no local imports) |

## CLI Flags

All hyperparameters can be overridden via `--key value` flags on any script:

```bash
# Training with custom params
python train.py --phase1_epochs 5 --lr 1e-4 --self_play_rounds 3

# Quick sanity check with overrides
python sanity_checks.py --phase1_epochs 1 --self_play_rounds 0

# Inference on CPU with limited validation
python inference.py --device cpu --val_words 500

# Change data location (e.g. Kaggle)
python train.py --data_dir /kaggle/input/meltwater-hangman
```

| Flag | Default | Description |
|------|---------|-------------|
| `--phase1_epochs` | 3 | Number of Phase 1 (MC supervised) epochs |
| `--self_play_rounds` | 2 | Number of Phase 2 self-play rounds |
| `--self_play_epochs_per_round` | 1 | Epochs per self-play round |
| `--lr` | 3e-4 | Learning rate |
| `--batch_size` | 512 | Training batch size |
| `--val_batch_size` | 4096 | Validation/inference batch size |
| `--max_train_steps` | None | Cap on optimizer steps per epoch |
| `--max_mc_words_per_epoch` | 0 | Limit MC words per epoch (0 = all) |
| `--dropout` | 0.1 | Model dropout |
| `--model_dim` | 256 | Transformer hidden dimension |
| `--num_layers` | 4 | Transformer encoder layers |
| `--num_heads` | 4 | Attention heads |
| `--correct_guess_prob` | 0.40 | MC correct-letter bias |
| `--mc_mix_ratio` | 0.30 | Self-play MC mix fraction |
| `--val_split_ratio` | 0.10 | Validation split fraction |
| `--device` | cuda/cpu | Device to run on |
| `--data_dir` | ./data | Path to data directory |
| `--seed` | 1920 | Random seed |

## Local Testing

```bash
# Activate the venv
& "D:\Sahil\mystuff\Hackathons\KLA_hackathon\venv\Scripts\python.exe"

# Run all sanity checks (~2 min on CPU)
& "D:\Sahil\mystuff\Hackathons\KLA_hackathon\venv\Scripts\python.exe" sanity_checks.py

# Full training (requires GPU, ~30 min)
& "D:\Sahil\mystuff\Hackathons\KLA_hackathon\venv\Scripts\python.exe" train.py

# Generate submission from checkpoint
& "D:\Sahil\mystuff\Hackathons\KLA_hackathon\venv\Scripts\python.exe" inference.py
```

## Kaggle Submission

1. Upload `notebook.ipynb` as a new Kaggle notebook
2. Add `meltwater-hangman` as a dataset input (containing `train.txt`, `test.txt`, `sample_submission.csv`)
3. Set runtime to GPU
4. Run all cells — generates `submission.csv` in the working directory
5. Submit `submission.csv`

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- tqdm

No `transformers` library, no pretrained weights, no internet access required.
