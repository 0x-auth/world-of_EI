# genesis-intelligence

A minimal implementation of the "born with memory, not trained then born" thesis from the BAZINGA-INTELLIGENCE-ROADMAP.

## The idea, plainly

Standard AI: train on a huge corpus first, then it's "born" and starts responding.

This: born empty, experiences one thing at a time, remembers what **changed** (not the raw input), and reconstructs an account of its own past when asked — instead of doing a database lookup. Its behavior shifts based on which changes repeat. There is no separate training phase, ever.

## What's actually here (and what isn't)

Kept, because it's load-bearing:
- **Delta-based memory**: stores what changed, not raw snapshots
- **Reconstruction, not retrieval**: querying the system rebuilds an answer from its lived history each time, rather than fetching a stored fact
- **Learning as pattern frequency**: what the system is "familiar with" is just a count of repeated changes over its own lived experience — no separate training step
- **Persistence**: memory survives across sessions (save/load), proven by a reload test in the demo

Deliberately left out, because the BAZINGA corpus review found them decorative, not load-bearing:
- Golden-ratio harmonics, 432Hz constants, the 5-dimension DODO framework
- Quantum-linguistics benchmarks (no real methodology behind the numbers in the original corpus)

## Run it

```bash
python3 genesis.py
```

Demonstrates: a fresh instance experiencing a few state changes, reconstructing its own history on query, reporting its most familiar patterns, saving to disk, then reloading as a genuinely new process and proving the memory came back intact.

## Why this is small on purpose

This is a proof of the *mechanism*, not a claim of intelligence. It shows the delta-memory-reconstruction loop actually runs and actually persists — which is the specific thing the 2024 BAZINGA prototype's own test results flagged as broken (no state persistence, no long-term memory). Everything past this point — richer pattern recognition, actual language generation, more sophisticated learning — is a real next step, not yet built here.

## Origin

Distilled from `0x-auth/BAZINGA` after a full read of its ~76 markdown docs. Most of that corpus is either decorative framing or unrelated content; this repo keeps only the one mechanism that held up under scrutiny.
