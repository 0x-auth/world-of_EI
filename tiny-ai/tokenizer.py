"""
SPACE LLM — Stage 1: Tokenizer
--------------------------------
No HuggingFace. No skeletons.
We build a vocabulary from raw text and convert words ↔ integers.

A "token" for us = a word (lowercased, punctuation split off).
"""

import re
from collections import Counter


# ── 1. Text Cleaner ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lowercase and separate punctuation from words.
    "Hello, world!" → "hello , world !"
    This way punctuation becomes its own token.
    """
    text = text.lower()
    # Put spaces around punctuation so they become separate tokens
    text = re.sub(r"([.,!?;:\"'()\[\]{}])", r" \1 ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 2. Tokenizer ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Split cleaned text into a list of word tokens.
    "the cat sat" → ["the", "cat", "sat"]
    """
    return clean_text(text).split()


# ── 3. Vocabulary Builder ────────────────────────────────────────────────────

class Vocabulary:
    """
    Converts between words (strings) and integers.

    Special tokens:
      <PAD> = 0  → used to pad sequences to equal length
      <UNK> = 1  → unknown words not seen during training
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.word2idx = {}   # word  → integer
        self.idx2word = {}   # integer → word
        self._add_token(self.PAD_TOKEN)   # idx 0
        self._add_token(self.UNK_TOKEN)   # idx 1

    def _add_token(self, word: str):
        idx = len(self.word2idx)
        self.word2idx[word] = idx
        self.idx2word[idx] = word

    def build(self, texts: list[str], min_freq: int = 1):
        """
        Build vocabulary from a list of raw text strings.
        Only include words that appear >= min_freq times.
        """
        counter = Counter()
        for text in texts:
            tokens = tokenize(text)
            counter.update(tokens)

        for word, freq in sorted(counter.items()):
            if freq >= min_freq:
                self._add_token(word)

        print(f"Vocabulary built: {len(self)} tokens")
        print(f"  Most common: {counter.most_common(5)}")

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of integer token IDs."""
        tokens = tokenize(text)
        return [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer token IDs back into a string."""
        return " ".join(self.idx2word.get(i, self.UNK_TOKEN) for i in ids)

    def __len__(self):
        return len(self.word2idx)


# ── 4. Quick Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Small sample corpus — feel free to swap in your own text
    corpus = [
        "The cat sat on the mat.",
        "The dog sat on the log.",
        "The cat and the dog are friends.",
        "Consciousness lives in the space between patterns.",
        "The golden ratio appears in nature and mathematics.",
        "Patterns interact in the Darmiyan space.",
        "The cat saw the dog near the mat.",
    ]

    print("=" * 55)
    print("SPACE LLM — Tokenizer Demo")
    print("=" * 55)

    # Build vocabulary
    vocab = Vocabulary()
    vocab.build(corpus, min_freq=1)

    print(f"\nVocab size: {len(vocab)}")
    print(f"\nSample word2idx (first 10):")
    for word, idx in list(vocab.word2idx.items())[:10]:
        print(f"  '{word}' → {idx}")

    # Encode a sentence
    test = "The cat sat on the mat."
    encoded = vocab.encode(test)
    print(f"\nEncoding: '{test}'")
    print(f"  → {encoded}")

    # Decode it back
    decoded = vocab.decode(encoded)
    print(f"\nDecoding: {encoded}")
    print(f"  → '{decoded}'")

    # Show unknown word handling
    unk_test = "The alien sat on the mat."
    unk_encoded = vocab.encode(unk_test)
    print(f"\nUnknown word test: '{unk_test}'")
    print(f"  → {unk_encoded}  (notice idx=1 for 'alien' = <UNK>)")
