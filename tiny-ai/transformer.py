"""
SPACE LLM — Stage 4: Transformer Block
----------------------------------------
No HuggingFace. No skeletons.

One transformer block contains:
  1. LayerNorm
  2. Multi-Head Attention  (from stage 3)
  3. Residual connection
  4. LayerNorm
  5. Feed Forward Network
  6. Residual connection

Then we stack N of these blocks to make the full model backbone.
Finally we add a linear "language model head" that projects
from embed_dim → vocab_size to produce next-word probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from attention import MultiHeadAttention, make_causal_mask


# ── 1. Feed Forward Network ───────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two linear layers with GELU activation between them.

      x → Linear(embed_dim, 4*embed_dim) → GELU → Linear(4*embed_dim, embed_dim)

    The 4x expansion is standard (from "Attention is All You Need").
    It gives the model a larger space to "think" in between attention steps.

    GELU vs ReLU: GELU is smoother — it doesn't hard-zero negatives,
    just softly suppresses them. Works better in practice for transformers.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),   # expand
            nn.GELU(),                              # non-linearity
            nn.Linear(4 * embed_dim, embed_dim),   # contract back
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 2. Single Transformer Block ───────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One full transformer block — the repeating unit of every LLM.

    Architecture (Pre-LN variant — more stable than original paper):

      x → LayerNorm → MultiHeadAttention → + x  (residual)
        → LayerNorm → FeedForward        → + x  (residual)

    The residual connections (+ x) are the key to training deep networks.
    They create a "highway" for gradients to flow back without vanishing.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = FeedForward(embed_dim, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Input:  x  shape (batch, seq_len, embed_dim)
        Output:    shape (batch, seq_len, embed_dim)
        """
        # Sub-layer 1: attention with residual
        # LayerNorm BEFORE attention (Pre-LN) — more stable gradients
        attn_out, _ = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop(attn_out)     # residual connection

        # Sub-layer 2: feed forward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out                  # residual connection

        return x


# ── 3. Full Model (stack of blocks + LM head) ─────────────────────────────────

class SpaceLLM(nn.Module):
    """
    The complete tiny language model.

    Architecture:
      InputEmbedding
        ↓
      N × TransformerBlock
        ↓
      LayerNorm
        ↓
      Linear (embed_dim → vocab_size)   ← the "LM head"
        ↓
      logits over vocabulary

    The LM head produces one score per vocabulary word for each position.
    The highest score = the model's predicted next word.

    We do NOT apply softmax here — we return raw logits.
    (CrossEntropyLoss expects logits, not probabilities.)
    """

    def __init__(self,
                 vocab_size:  int,
                 embed_dim:   int   = 64,
                 num_heads:   int   = 4,
                 num_layers:  int   = 2,
                 max_seq_len: int   = 128,
                 dropout:     float = 0.1):
        super().__init__()

        # Store config
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.num_layers  = num_layers
        self.max_seq_len = max_seq_len

        # Token + positional embedding
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = nn.Embedding(max_seq_len, embed_dim)
        # (Using learned positional embeddings here — simpler than sin/cos for training)
        self.drop      = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final normalisation
        self.norm = nn.LayerNorm(embed_dim)

        # LM head: project to vocabulary size
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and LM head
        # This is standard practice — reduces parameters, often improves results
        # Intuition: the embedding that represents "cat" as input
        #            should be similar to the output weight for predicting "cat"
        self.lm_head.weight = self.token_emb.weight

        # Initialise weights
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"SpaceLLM initialised — {total_params:,} parameters")

    def _init_weights(self):
        """Small random initialisation — prevents early saturation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Input:  token_ids  shape (batch, seq_len)
        Output: logits     shape (batch, seq_len, vocab_size)

        logits[b][i] = scores for all vocab words as the next token after position i
        """
        batch, seq_len = token_ids.shape
        assert seq_len <= self.max_seq_len, \
            f"Sequence too long: {seq_len} > {self.max_seq_len}"

        # Token embeddings
        tok = self.token_emb(token_ids)                          # (batch, seq, embed)

        # Positional embeddings — learned, not sin/cos
        positions = torch.arange(seq_len, device=token_ids.device)
        pos = self.pos_emb(positions)                            # (seq, embed)

        # Combine and dropout
        x = self.drop(tok + pos)                                 # (batch, seq, embed)

        # Causal mask — prevent attending to future tokens
        mask = make_causal_mask(seq_len).to(token_ids.device)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Final norm
        x = self.norm(x)                                         # (batch, seq, embed)

        # Project to vocabulary
        logits = self.lm_head(x)                                 # (batch, seq, vocab_size)

        return logits

    def count_params(self) -> dict:
        """Break down parameters by component."""
        counts = {}
        counts["token_emb"]  = self.token_emb.weight.numel()
        counts["pos_emb"]    = self.pos_emb.weight.numel()
        counts["blocks"]     = sum(p.numel() for b in self.blocks for p in b.parameters())
        counts["norm+head"]  = sum(p.numel() for p in self.norm.parameters())
        # lm_head shares weights with token_emb so not double-counted
        counts["total"]      = sum(p.numel() for p in self.parameters())
        return counts


# ── 4. Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tokenizer import Vocabulary

    print("=" * 55)
    print("SPACE LLM — Transformer Block Demo")
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

    vocab = Vocabulary()
    vocab.build(corpus)

    # Tiny model — readable but real
    model = SpaceLLM(
        vocab_size  = len(vocab),   # 28
        embed_dim   = 32,           # small but functional
        num_heads   = 2,
        num_layers  = 2,
        max_seq_len = 64,
        dropout     = 0.0,          # off for demo
    )

    # ── Parameter breakdown ──
    print("\nParameter breakdown:")
    for k, v in model.count_params().items():
        print(f"  {k:<15} {v:>8,}")

    # ── Forward pass ──
    sentence = "the cat sat on the mat"
    ids = vocab.encode(sentence)
    x = torch.tensor([ids])

    print(f"\nInput : '{sentence}'")
    print(f"Tokens: {ids}  shape={x.shape}")

    with torch.no_grad():
        logits = model(x)

    print(f"Logits: {logits.shape}  → (batch, seq_len, vocab_size)")

    # ── What does the model predict after "the cat sat" ? ──
    # logits[0][2] = scores for all words after seeing "the cat sat"
    position = 2   # after "sat" (index 2)
    scores   = logits[0, position]                      # (vocab_size,)
    probs    = torch.softmax(scores, dim=-1)             # probabilities
    top5_idx = torch.topk(probs, 5).indices.tolist()

    print(f"\nAfter seeing '{' '.join(sentence.split()[:position+1])}'")
    print(f"Top 5 predicted next words (untrained — random weights):")
    for idx in top5_idx:
        word = vocab.idx2word[idx]
        prob = probs[idx].item()
        print(f"  '{word}' → {prob:.4f} ({prob*100:.1f}%)")

    print(f"\nNote: these are random — model not trained yet.")
    print(f"After training, 'on' should be near the top.")
    print(f"\n✓ Stage 4 complete. Full forward pass works end to end.")
    print(f"  text → tokens → embeddings → attention → FFN → logits")
