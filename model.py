"""Self-contained CANINE-S style character Transformer for Hangman.

This re-implements the *architecture* described by the reference
``Hangman-with-Transformers`` repo (a character-level transformer encoder
with a classification head over 26 letters) but WITHOUT any dependency on the
``transformers`` library, the CANINE tokenizer, or any pretrained weights.
Everything (embedding, positional encoding, transformer encoder, classifier)
is instantiated from random init with plain PyTorch so the submission remains
fully reproducible and offline on Kaggle.
"""
import math
import torch
import torch.nn as nn

import config as cfg


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() *
                             (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CanineHangmanModel(nn.Module):
    """Character-level transformer encoder consuming a Hangman state and
    emitting a [B, 26] logits tensor of next-letter scores."""

    def __init__(self, vocab_size=cfg.VOCAB_SIZE, dim=cfg.MODEL_DIM,
                 num_heads=cfg.NUM_HEADS, num_layers=cfg.NUM_LAYERS,
                 ff_dim=cfg.FF_DIM, dropout=cfg.DROPOUT,
                 max_len=cfg.MAX_SEQ_LEN, num_classes=cfg.NUM_LETTERS):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=cfg.PAD)
        self.pos_enc = PositionalEncoding(dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        # Pool on the CLS token (position 0) -> 26 letter logits
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, input_ids, attn_mask=None):
        """input_ids: [B, T] long; attn_mask: [B, T] long (1=real, 0=pad) or None.
        Returns logits [B, num_classes]."""
        B, T = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)
        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = attn_mask == 0  # True == pad to ignore
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        cls_hidden = x[:, 0]  # CLS token
        logits = self.classifier(self.dropout(cls_hidden))
        return logits


class HangmanAgent:
    """Thin wrapper around the model + prediction logic.

    predict(input_ids, attn_mask, guessed_mask) -> next letter id (0..25)
    Already-guessed letters are masked to -inf before argmax so repeats
    never occur.
    """

    def __init__(self, model=None):
        self.model = model if model is not None else CanineHangmanModel()
        self.device = next(self.model.parameters()).device

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    @torch.no_grad()
    def predict(self, input_ids, attn_mask, guessed_mask=None):
        """input_ids/attn_mask: tensors [B, T]; guessed_mask: [B, 26] bool.
        Returns next letter id tensor [B]."""
        logits = self.model(input_ids, attn_mask)  # [B, 26]
        if guessed_mask is not None:
            logits = logits.masked_fill(guessed_mask, float("-inf"))
        next_id = logits.argmax(dim=-1)  # [B]
        # Safety fallback for the pathological all-masked case
        if guessed_mask is not None:
            all_guessed = guessed_mask.all(dim=-1)
            next_id = torch.where(all_guessed, torch.zeros_like(next_id), next_id)
        return next_id
