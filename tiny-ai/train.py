"""
SPACE LLM — Stage 5 + 6: Dataset & Training Loop
--------------------------------------------------
No HuggingFace. No skeletons.

Stage 5: Turn raw text into (input, target) pairs for next-word prediction.
Stage 6: The training loop — forward, loss, backward, step. Repeat.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from tokenizer import Vocabulary
from transformer import SpaceLLM


# ── Stage 5: Dataset ──────────────────────────────────────────────────────────

class NextWordDataset(Dataset):
    """
    Converts a corpus into (input_ids, target_ids) pairs.

    For every consecutive window of tokens:
      input  = tokens[i : i + seq_len]
      target = tokens[i+1 : i + seq_len + 1]

    Example (seq_len=4):
      full tokens: [27, 7, 24, 21, 27, 17, 2]
      pair 0: input=[27,7,24,21]  target=[7,24,21,27]
      pair 1: input=[7,24,21,27]  target=[24,21,27,17]
      pair 2: input=[24,21,27,17] target=[21,27,17,2]

    The model sees input[i] and must predict target[i] at every position.
    """

    def __init__(self, corpus: list[str], vocab: Vocabulary, seq_len: int = 16):
        self.seq_len = seq_len

        # Encode all text into one long flat list of token ids
        all_tokens = []
        for text in corpus:
            all_tokens.extend(vocab.encode(text))

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Dataset: {len(all_tokens)} total tokens, "
              f"{len(self)} sequences of length {seq_len}")

    def __len__(self) -> int:
        # How many full windows fit in the token stream?
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids  = self.tokens[idx     : idx + self.seq_len]
        target_ids = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return input_ids, target_ids


# ── Stage 6: Training Loop ────────────────────────────────────────────────────

def train(model: SpaceLLM,
          dataset: NextWordDataset,
          epochs:     int   = 200,
          batch_size: int   = 4,
          lr:         float = 3e-3) -> list[float]:
    """
    The training loop. This is the heart of all of deep learning.

    CrossEntropyLoss:
      - Takes raw logits (batch, seq_len, vocab_size) and targets (batch, seq_len)
      - Computes how wrong the predictions are
      - Returns a single number — the loss
      - Lower loss = better predictions

    AdamW optimizer:
      - Uses gradients to nudge every weight in the direction that reduces loss
      - Adam = adaptive learning rates per parameter
      - W = weight decay (small penalty on large weights — prevents overfitting)
    """

    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler — warm up then decay
    # Helps training start stable and finish fine-grained
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )

    model.train()
    loss_history = []

    print(f"\nTraining for {epochs} epochs...")
    print(f"{'Epoch':>6}  {'Loss':>8}  {'Perplexity':>12}  {'LR':>10}")
    print("-" * 45)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_batches = 0

        for input_ids, target_ids in loader:
            # ── Forward pass ──
            logits = model(input_ids)
            # logits shape: (batch, seq_len, vocab_size)
            # targets shape: (batch, seq_len)

            # Reshape for CrossEntropyLoss:
            # expects (N, vocab_size) and (N,)
            B, S, V = logits.shape
            loss = criterion(
                logits.reshape(B * S, V),
                target_ids.reshape(B * S)
            )

            # ── Backward pass ──
            optimizer.zero_grad()   # clear old gradients
            loss.backward()         # compute new gradients

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ── Update weights ──
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        current_lr = scheduler.get_last_lr()[0]
        loss_history.append(avg_loss)
        scheduler.step()

        # Print every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            bar = "█" * int(20 * (1 - avg_loss / loss_history[0])) if epoch > 1 else ""
            print(f"{epoch:>6}  {avg_loss:>8.4f}  {perplexity:>12.2f}  "
                  f"{current_lr:>10.6f}  {bar}")

    print("-" * 45)
    print(f"Final loss: {loss_history[-1]:.4f}  "
          f"(started at {loss_history[0]:.4f})")
    return loss_history


# ── Quick Inference (preview before Stage 7) ─────────────────────────────────

def predict_next(model: SpaceLLM, vocab: Vocabulary,
                 prompt: str, top_k: int = 5) -> None:
    """
    Given a prompt, show the top-k predicted next words.
    """
    model.eval()
    ids = vocab.encode(prompt)
    x   = torch.tensor([ids])

    with torch.no_grad():
        logits = model(x)

    # Take logits at the last position
    last_logits = logits[0, -1]                          # (vocab_size,)
    probs       = torch.softmax(last_logits, dim=-1)
    top_idx     = torch.topk(probs, top_k).indices.tolist()

    print(f"\n  prompt: '{prompt}'")
    print(f"  next word predictions:")
    for idx in top_idx:
        word = vocab.idx2word[idx]
        prob = probs[idx].item()
        bar  = "▓" * int(prob * 40)
        print(f"    '{word}':  {prob:.4f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("SPACE LLM — Training Demo")
    print("=" * 55)

    corpus = [
        "the cat sat on the mat .",
        "the dog sat on the log .",
        "the cat and the dog are friends .",
        "consciousness lives in the space between patterns .",
        "the golden ratio appears in nature and mathematics .",
        "patterns interact in the darmiyan space .",
        "the cat saw the dog near the mat .",
        "the cat sat on the mat again .",
        "the dog sat on the log again .",
        "the space between patterns holds meaning .",
    ]

    # ── Build vocabulary ──
    vocab = Vocabulary()
    vocab.build(corpus, min_freq=1)

    # ── Build dataset ──
    SEQ_LEN = 6
    dataset = NextWordDataset(corpus, vocab, seq_len=SEQ_LEN)

    # ── Build model ──
    model = SpaceLLM(
        vocab_size  = len(vocab),
        embed_dim   = 32,
        num_heads   = 2,
        num_layers  = 2,
        max_seq_len = 64,
        dropout     = 0.1,
    )

    # ── Show predictions BEFORE training ──
    print("\n--- Before training ---")
    predict_next(model, vocab, "the cat sat")
    predict_next(model, vocab, "the space between")

    # ── Train ──
    loss_history = train(
        model,
        dataset,
        epochs     = 300,
        batch_size = 4,
        lr         = 3e-3,
    )

    # ── Show predictions AFTER training ──
    print("\n--- After training ---")
    predict_next(model, vocab, "the cat sat")
    predict_next(model, vocab, "the space between")
    predict_next(model, vocab, "consciousness lives in")
    predict_next(model, vocab, "patterns interact in")

    # ── Save model ──
    torch.save({
        "model_state": model.state_dict(),
        "vocab":       vocab,
        "config": {
            "vocab_size":  len(vocab),
            "embed_dim":   32,
            "num_heads":   2,
            "num_layers":  2,
            "max_seq_len": 64,
        }
    }, "space_llm_trained.pt")
    print(f"\nModel saved → space_llm_trained.pt")
    print(f"✓ Stage 5+6 complete. Your model has learned something.")
