# Logic = knowledge + squeeze (grokking, a×b mod 97)

The test cited in DEFINITIONS.md for **logic** [LOGIC].

Two identical networks learn a×b mod 97 from 40% of the table (3,763 pairs).
The other 5,646 pairs are never shown. One network is squeezed (weight decay 1.0), one is not.

| | seen pairs | unseen pairs | weight norm |
|---|---|---|---|
| squeezed, step 5,000 | 100% | **99.9%** | 158 → 82 |
| unsqueezed, step 8,000 | 100% | **2%** | 158 → 269 |

Same knowledge; only the squeezed one finds the rule. Without the squeeze: a lookup table.

Run from the repo root (needs torch):

    python3 experiments/grokking/grokking_experiment.py 1.0 8000   # squeezed
    python3 experiments/grokking/grokking_experiment.py 0.0 8000   # unsqueezed

Logs go to results/grokking_log_wd<WD>.json. Seed 0, about 30 s per 1,000 steps on CPU.
