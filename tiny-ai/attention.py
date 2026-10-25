"""
SPACE LLM — Stage 3: Attention Mechanism
------------------------------------------
No HuggingFace. No skeletons.

We build two things:
  1. SingleHeadAttention  — the pure attention equation, naked
  2. MultiHeadAttention   — run attention H times in parallel, then combine

Why multi-head?
  One "head" can only learn one kind of relationship at a time.
  Multiple heads let the model attend to syntax AND meaning AND position
  simultaneously — each head specialises on its own.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── 1. Single Head Attention ──────────────────────────────────────────────────

class SingleHeadAttention(nn.Module):
    """
    Pure implementation of:
      Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) * V

    Three learned linear projections turn the input into Q, K, V.
    That's the only "learning" here — the projection weights.
    The attention equation itself is fixed math.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.d_k = embed_dim

        # Three weight matrices — the only learned parameters in attention
        # Each projects input (embed_dim) → query/key/value (embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)  # query projection
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)  # key projection
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)  # value projection

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:  x     shape (batch, seq_len, embed_dim)
                mask  shape (batch, seq_len, seq_len)  — optional causal mask

        Output: out   shape (batch, seq_len, embed_dim)
                attn  shape (batch, seq_len, seq_len)  — attention weights (for viz)
        """

        # Step 1: Project input into Q, K, V
        # Each is shape (batch, seq_len, embed_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Compute raw attention scores
        # Q shape:  (batch, seq_len, d_k)
        # K^T shape:(batch, d_k, seq_len)
        # scores:   (batch, seq_len, seq_len)
        # scores[b][i][j] = "how much should word i attend to word j?"
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Step 3: Scale by sqrt(d_k)
        # Without this, dot products grow large with dimension → softmax saturates
        # (gradients vanish, nothing learns)
        scores = scores / math.sqrt(self.d_k)

        # Step 4: Apply causal mask (if provided)
        # In a language model, word at position i must NOT see words at i+1, i+2...
        # (we can't peek at future words when predicting the next one)
        # We set those positions to -infinity so softmax makes them ~0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 5: Softmax → attention weights (probabilities, sum to 1)
        # Shape: (batch, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)

        # Step 6: Weighted sum of values
        # attn_weights: (batch, seq_len, seq_len)
        # V:            (batch, seq_len, embed_dim)
        # output:       (batch, seq_len, embed_dim)
        out = torch.matmul(attn_weights, V)

        return out, attn_weights


# ── 2. Causal Mask Builder ────────────────────────────────────────────────────

def make_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Build a lower-triangular mask of shape (seq_len, seq_len).

    Position i can attend to positions 0..i only (not the future).

    Example for seq_len=4:
      [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [1, 1, 1, 0],
       [1, 1, 1, 1]]

    Zeros become -inf in the attention scores → ~0 after softmax.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


# ── 3. Multi-Head Attention ───────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Run attention h times in parallel, each in a smaller subspace.

    Instead of one attention over full embed_dim:
      - Split embed_dim into h chunks of size (embed_dim / h)
      - Run attention independently on each chunk (each is a "head")
      - Concatenate all head outputs
      - Final linear projection to mix them back together

    Why? Each head can specialise:
      Head 1 → learns syntactic relationships (subject-verb)
      Head 2 → learns semantic relationships (cat-mat rhyme?)
      Head 3 → learns positional patterns (nearby words)
      ...etc
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head    = embed_dim // num_heads   # dimension per head

        # One big projection for all heads at once (efficient)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Final projection after concatenating all heads
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:  x  shape (batch, seq_len, embed_dim)
        Output: (batch, seq_len, embed_dim), attention weights
        """
        batch, seq_len, _ = x.shape

        # Step 1: Project and split into heads
        # After linear: (batch, seq_len, embed_dim)
        # After reshape: (batch, seq_len, num_heads, d_head)
        # After transpose: (batch, num_heads, seq_len, d_head)
        Q = self.W_q(x).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # Step 2: Scaled dot-product attention — same equation, now per head
        # scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Step 3: Apply causal mask
        if mask is not None:
            # Expand mask for batch and heads dimensions
            m = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(m == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        # Step 4: Weighted sum of values
        # (batch, num_heads, seq_len, d_head)
        out = torch.matmul(attn_weights, V)

        # Step 5: Concatenate heads back together
        # transpose → (batch, seq_len, num_heads, d_head)
        # reshape   → (batch, seq_len, embed_dim)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)

        # Step 6: Final output projection
        out = self.W_o(out)

        return out, attn_weights


# ── 4. Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from tokenizer import Vocabulary
    from embeddings import InputEmbedding

    print("=" * 55)
    print("SPACE LLM — Attention Demo")
    print("=" * 55)

    corpus = [
        "The cat sat on the mat.",
        "The dog sat on the log.",
        "The cat and the dog are friends.",
        "Consciousness lives in the space between patterns.",
        "The golden ratio appears in nature and mathematics.",
        "Patterns interact in the Darmiyan space.",
        "The cat saw the dog near the mat.",
    ]

    EMBED_DIM  = 16
    NUM_HEADS  = 2   # 16 / 2 = 8 dims per head

    vocab = Vocabulary()
    vocab.build(corpus)
    embedder = InputEmbedding(len(vocab), EMBED_DIM, dropout=0.0)

    # Encode a sentence
    sentence = "the cat sat on the mat"
    ids = vocab.encode(sentence)
    x = torch.tensor([ids])          # shape (1, 6)

    with torch.no_grad():
        embedded = embedder(x)       # shape (1, 6, 16)

    print(f"\nSentence : '{sentence}'")
    print(f"Tokens   : {ids}")
    print(f"Embedded : {embedded.shape}  → (batch, seq_len, embed_dim)")

    # ── Single head (no mask) ──
    print("\n--- Single Head Attention (no mask) ---")
    sha = SingleHeadAttention(EMBED_DIM)
    with torch.no_grad():
        out_sha, attn_sha = sha(embedded)

    print(f"Output shape : {out_sha.shape}")
    print(f"Attn weights : {attn_sha.shape}  → (batch, seq_len, seq_len)")
    print(f"\nAttention matrix (how much each word attends to each other):")
    words = sentence.split()
    print(f"  {'':8}", end="")
    for w in words: print(f"{w:8}", end="")
    print()
    for i, w in enumerate(words):
        print(f"  {w:8}", end="")
        for j in range(len(words)):
            print(f"{attn_sha[0, i, j].item():8.3f}", end="")
        print()

    # ── Causal mask ──
    print("\n--- Causal Mask (language model — can't see future) ---")
    mask = make_causal_mask(len(ids))
    print(f"Mask for seq_len={len(ids)}:")
    print(mask.int().numpy())

    # ── Multi-head with causal mask ──
    print("\n--- Multi-Head Attention (2 heads, with causal mask) ---")
    mha = MultiHeadAttention(EMBED_DIM, NUM_HEADS)
    with torch.no_grad():
        out_mha, attn_mha = mha(embedded, mask=mask)

    print(f"Output shape      : {out_mha.shape}")
    print(f"Attn weights shape: {attn_mha.shape}  → (batch, heads, seq, seq)")
    print(f"\nHead 0 attention (first head's view):")
    print(f"  {'':8}", end="")
    for w in words: print(f"{w:8}", end="")
    print()
    for i, w in enumerate(words):
        print(f"  {w:8}", end="")
        for j in range(len(words)):
            v = attn_mha[0, 0, i, j].item()
            print(f"{v:8.3f}", end="")
        print()

    print(f"\n✓ Stage 3 complete. Words now understand their context.")
    print(f"  'cat' now knows it's the one 'sat' — not a president.")
