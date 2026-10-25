"""
SPACE LLM — Stage 7: Inference
--------------------------------
No HuggingFace. No skeletons.

This is where the model generates text — word by word, feeding
each prediction back as the next input.

We implement three generation strategies:
  1. Greedy     — always pick the highest probability word
  2. Top-k      — sample randomly from the k most likely words
  3. Temperature — scale logits before softmax to control randomness

Greedy = deterministic but repetitive
Top-k + temperature = creative but sometimes incoherent
"""

import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from tokenizer import Vocabulary
from transformer import SpaceLLM


# ── 1. Load trained model ─────────────────────────────────────────────────────

def load_model(path: str) -> tuple[SpaceLLM, Vocabulary]:
    """Load the saved model and vocabulary from disk."""
    checkpoint = torch.load(path, weights_only=False)
    cfg   = checkpoint["config"]
    vocab = checkpoint["vocab"]

    model = SpaceLLM(
        vocab_size  = cfg["vocab_size"],
        embed_dim   = cfg["embed_dim"],
        num_heads   = cfg["num_heads"],
        num_layers  = cfg["num_layers"],
        max_seq_len = cfg["max_seq_len"],
        dropout     = 0.0,   # no dropout during inference
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded from '{path}'")
    print(f"Vocabulary: {len(vocab)} tokens")
    return model, vocab


# ── 2. Greedy Generation ──────────────────────────────────────────────────────

def generate_greedy(model: SpaceLLM, vocab: Vocabulary,
                    prompt: str, max_new_words: int = 10) -> str:
    """
    Always pick the single highest-probability next word.

    Loop:
      1. encode current sequence
      2. forward pass → logits
      3. take logits at last position
      4. argmax → highest scoring word
      5. append to sequence
      6. repeat

    Deterministic — same prompt always gives same output.
    Tends to get stuck in loops for long sequences.
    """
    ids = vocab.encode(prompt)

    for _ in range(max_new_words):
        x      = torch.tensor([ids])
        with torch.no_grad():
            logits = model(x)                       # (1, seq, vocab)

        next_logits = logits[0, -1]                 # scores for next word
        next_id     = torch.argmax(next_logits).item()
        next_word   = vocab.idx2word[next_id]

        ids.append(next_id)

        if next_word == ".":                        # stop at sentence end
            break

    return vocab.decode(ids)


# ── 3. Temperature + Top-k Sampling ──────────────────────────────────────────

def generate_sample(model: SpaceLLM, vocab: Vocabulary,
                    prompt: str,
                    max_new_words: int = 15,
                    temperature: float = 1.0,
                    top_k: int = 5) -> str:
    """
    Sample from the probability distribution instead of always picking the top.

    Temperature:
      < 1.0  → sharper distribution → more confident, less creative
        1.0  → unchanged distribution
      > 1.0  → flatter distribution → more random, more creative

      How: divide logits by temperature before softmax
        high temp → logits closer together → more uniform probs
        low temp  → logits further apart   → winner takes all

    Top-k:
      Only sample from the k most likely words.
      Prevents the model picking truly nonsense low-probability words.
    """
    ids = vocab.encode(prompt)

    for _ in range(max_new_words):
        x = torch.tensor([ids])
        with torch.no_grad():
            logits = model(x)

        next_logits = logits[0, -1]                 # (vocab_size,)

        # Apply temperature
        next_logits = next_logits / temperature

        # Top-k filtering: zero out everything except top k
        if top_k > 0:
            values, indices = torch.topk(next_logits, top_k)
            # Create a mask — everything NOT in top-k becomes -inf
            filtered = torch.full_like(next_logits, float("-inf"))
            filtered.scatter_(0, indices, values)
            next_logits = filtered

        # Sample from distribution
        probs   = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        next_word = vocab.idx2word[next_id]
        ids.append(next_id)

        if next_word == ".":
            break

    return vocab.decode(ids)


# ── 4. Beam Search ────────────────────────────────────────────────────────────

def generate_beam(model: SpaceLLM, vocab: Vocabulary,
                  prompt: str,
                  max_new_words: int = 10,
                  beam_width: int = 3) -> str:
    """
    Keep the top `beam_width` candidate sequences at each step.

    Greedy only keeps 1 candidate — it might miss a better path.
    Beam search keeps beam_width candidates and picks the best overall.

    Example with beam_width=3:
      Step 1: 'the cat sat' → top 3: ['on', 'near', 'by']
      Step 2 from 'on':   → top 3: ['the', 'a', 'my']
      Step 2 from 'near': → top 3: ['the', 'a', 'that']
      Step 2 from 'by':   → top 3: ['the', 'a', 'his']
      Keep top 3 overall by cumulative log probability
      ...
    """
    ids = vocab.encode(prompt)

    # Each beam: (cumulative_log_prob, token_id_list)
    beams = [(0.0, ids[:])]

    for _ in range(max_new_words):
        candidates = []

        for score, beam_ids in beams:
            # Check if this beam already ended
            if beam_ids and vocab.idx2word[beam_ids[-1]] == ".":
                candidates.append((score, beam_ids))
                continue

            x = torch.tensor([beam_ids])
            with torch.no_grad():
                logits = model(x)

            next_logits = logits[0, -1]
            log_probs   = F.log_softmax(next_logits, dim=-1)

            # Take top beam_width next tokens
            top_lp, top_idx = torch.topk(log_probs, beam_width)

            for lp, idx in zip(top_lp.tolist(), top_idx.tolist()):
                candidates.append((score + lp, beam_ids + [idx]))

        # Keep only the top beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

        # Stop if all beams ended with "."
        if all(vocab.idx2word[b[-1]] == "." for _, b in beams):
            break

    best_ids = beams[0][1]
    return vocab.decode(best_ids)


# ── 5. Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("SPACE LLM — Inference Demo")
    print("=" * 55)

    # Load trained model
    model, vocab = load_model("space_llm_trained.pt")

    prompts = [
        "the cat sat",
        "the space between",
        "consciousness lives",
        "patterns interact",
        "the golden",
    ]

    # ── Strategy 1: Greedy ──
    print("\n" + "─" * 55)
    print("STRATEGY 1 — Greedy (always pick highest prob)")
    print("─" * 55)
    for p in prompts:
        out = generate_greedy(model, vocab, p, max_new_words=8)
        print(f"  '{p}' →")
        print(f"    {out}")

    # ── Strategy 2: Sampling ──
    print("\n" + "─" * 55)
    print("STRATEGY 2 — Sampling (temperature=0.8, top_k=5)")
    print("─" * 55)
    torch.manual_seed(42)
    for p in prompts:
        out = generate_sample(model, vocab, p,
                              max_new_words=8,
                              temperature=0.8,
                              top_k=5)
        print(f"  '{p}' →")
        print(f"    {out}")

    # ── Strategy 3: Beam Search ──
    print("\n" + "─" * 55)
    print("STRATEGY 3 — Beam Search (beam_width=3)")
    print("─" * 55)
    for p in prompts:
        out = generate_beam(model, vocab, p,
                            max_new_words=8,
                            beam_width=3)
        print(f"  '{p}' →")
        print(f"    {out}")

    # ── Temperature experiment ──
    print("\n" + "─" * 55)
    print("TEMPERATURE EXPERIMENT — same prompt, different creativity")
    print("─" * 55)
    prompt = "the cat"
    for temp in [0.3, 0.7, 1.0, 1.5]:
        torch.manual_seed(99)
        out = generate_sample(model, vocab, prompt,
                              max_new_words=8,
                              temperature=temp,
                              top_k=8)
        label = "conservative" if temp < 0.7 else ("creative" if temp > 1.0 else "balanced")
        print(f"  temp={temp} ({label}):  {out}")

    print(f"\n{'─'*55}")
    print(f"✓ Stage 7 complete. Your LLM generates text.")
    print(f"\nFull pipeline you built from scratch:")
    print(f"  tokenizer  → vocabulary + encode/decode")
    print(f"  embeddings → words to vectors + positions")
    print(f"  attention  → words understand each other")
    print(f"  transformer→ stack of attention + FFN blocks")
    print(f"  dataset    → text into training pairs")
    print(f"  training   → forward, loss, backward, step")
    print(f"  inference  → greedy, sampling, beam search")
    print(f"\nSame architecture as GPT. Same training loop as Claude.")
    print(f"Just 28,256 parameters instead of hundreds of billions. 🌌")
