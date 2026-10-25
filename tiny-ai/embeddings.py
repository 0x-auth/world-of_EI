"""
SPACE LLM — Stage 2: Embeddings
---------------------------------
No HuggingFace. No skeletons.

Two things we build here:
  1. TokenEmbedding   — maps word IDs → dense vectors
  2. PositionalEncoding — tells the model WHERE each word is in the sentence
  3. InputEmbedding   — combines both into one final input

Why do we need this?
  The transformer only understands vectors of floats.
  This stage is the bridge: integers → meaningful vectors.
"""

import torch
import torch.nn as nn
import math


# ── 1. Token Embedding ────────────────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    """
    A simple lookup table.
    vocab_size rows, embed_dim columns.
    Each row = the learned vector for one word.

    Example with vocab_size=28, embed_dim=8:
      Row 7  = vector for "cat"   → [0.2, -0.5, 0.8, ...]
      Row 27 = vector for "the"   → [0.9,  0.1, 0.0, ...]

    These start random. Training will adjust them.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # nn.Embedding is just a matrix — (vocab_size x embed_dim)
        # We ARE allowed to use this: it's a single matrix, not a skeleton
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Input:  token_ids  shape (batch_size, seq_len)  — integers
        Output: vectors    shape (batch_size, seq_len, embed_dim) — floats
        """
        return self.embedding(token_ids)


# ── 2. Positional Encoding ────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Adds position information to each token vector.

    Without this, the model sees "cat sat mat" and "mat cat sat"
    as identical — same tokens, same vectors. That's wrong.

    We use a fixed mathematical pattern (not learned):
      PE(pos, 2i)   = sin(pos / 10000^(2i/d))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    This is the original formula from "Attention is All You Need" (2017).
    Even dimensions get sine, odd dimensions get cosine.
    Each position gets a unique fingerprint of sin/cos waves.

    Why sin/cos? Because:
      - They're bounded between -1 and 1 (stable)
      - Similar positions get similar encodings
      - The model can learn to attend to relative positions
    """

    def __init__(self, embed_dim: int, max_seq_len: int = 512):
        super().__init__()

        # Build the positional encoding matrix once, reuse forever
        # Shape: (max_seq_len, embed_dim)
        pe = torch.zeros(max_seq_len, embed_dim)

        # Position indices: 0, 1, 2, ..., max_seq_len-1
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # Frequency divisors for each dimension pair
        # Shape: (embed_dim/2,)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10000.0) / embed_dim)
        )

        # Even indices → sine
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd indices → cosine
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_seq_len, embed_dim)
        # So it broadcasts across any batch size
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter — won't be updated by optimizer)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x  shape (batch_size, seq_len, embed_dim)
        Output: x + positional encoding, same shape

        We slice pe to match the actual sequence length.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ── 3. Input Embedding (Token + Position combined) ───────────────────────────

class InputEmbedding(nn.Module):
    """
    Final input to the transformer = token embedding + positional encoding.

    Also applies dropout for regularization (prevents overfitting).
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_enc   = PositionalEncoding(embed_dim, max_seq_len)
        self.dropout   = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Input:  token_ids  shape (batch_size, seq_len)
        Output: vectors    shape (batch_size, seq_len, embed_dim)
        """
        # Scale embeddings by sqrt(embed_dim) — from the original paper
        # Keeps the positional encoding from dominating the token signal
        x = self.token_emb(token_ids) * math.sqrt(self.embed_dim)
        x = self.pos_enc(x)
        x = self.dropout(x)
        return x


# ── 4. Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from tokenizer import Vocabulary, tokenize

    print("=" * 55)
    print("SPACE LLM — Embeddings Demo")
    print("=" * 55)

    # ── Reuse our vocabulary from Stage 1 ──
    corpus = [
        "The cat sat on the mat.",
        "The dog sat on the log.",
        "The cat and the dog are friends.",
        "Consciousness lives in the space between patterns.",
        "The golden ratio appears in nature and mathematics.",
        "Patterns interact in the Darmiyan space.",
        "The cat saw the dog near the mat.",
    ]

    vocab = Vocabulary()
    vocab.build(corpus)

    VOCAB_SIZE = len(vocab)   # 28
    EMBED_DIM  = 16           # each word becomes a 16-float vector
    # (real models use 512, 768, or 4096 — we use 16 to keep it readable)

    # ── Build our embedding layer ──
    embedder = InputEmbedding(
        vocab_size  = VOCAB_SIZE,
        embed_dim   = EMBED_DIM,
        max_seq_len = 512,
        dropout     = 0.0   # off for demo so output is deterministic
    )

    # ── Encode a sentence into token IDs ──
    sentence = "the cat sat on the mat"
    token_ids = vocab.encode(sentence)
    print(f"\nSentence : '{sentence}'")
    print(f"Token IDs: {token_ids}")

    # ── Convert to tensor, add batch dimension ──
    # Shape: (1, seq_len) — batch of 1
    ids_tensor = torch.tensor([token_ids])
    print(f"\nInput tensor shape : {ids_tensor.shape}  (batch=1, seq_len={len(token_ids)})")

    # ── Run through embedder ──
    with torch.no_grad():
        output = embedder(ids_tensor)

    print(f"Output tensor shape: {output.shape}  (batch=1, seq_len={len(token_ids)}, embed_dim={EMBED_DIM})")

    # ── Show what "cat" looks like as a vector ──
    cat_idx  = token_ids.index(vocab.word2idx["cat"])
    cat_vec  = output[0, cat_idx, :]
    print(f"\n'cat' as a {EMBED_DIM}-dimensional vector:")
    print(f"  {[round(v, 3) for v in cat_vec.tolist()]}")

    # ── Show positional encoding effect ──
    # "the" appears twice (positions 0 and 4) — should have different vectors
    the_pos0 = output[0, 0, :]   # "the" at position 0
    the_pos4 = output[0, 4, :]   # "the" at position 4
    diff = (the_pos0 - the_pos4).abs().mean().item()
    print(f"\n'the' appears at position 0 and position 4.")
    print(f"Same word, different position → mean vector difference: {diff:.4f}")
    print(f"(Should be > 0 — positional encoding is working!)")

    print(f"\n✓ Stage 2 complete. Words are now vectors the transformer can read.")
