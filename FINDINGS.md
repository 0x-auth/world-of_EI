# EI — findings & where "worthy" actually lands

## What the original `ei.py` (v1) is
A bigram transition counter. For each line it records `context word(s) -> next word`
and predicts the most frequent next word. Persistent tape, predict->compare->scar->update,
learning that survives restarts — the EI *loop* is real and correct.

## What v1 cannot do (tested)
The README's own bar is: "related unseen material is handled better than before exposure."
v1 fails it. On a genuinely unseen file, tested:

    TRUE transfer to unseen file: accuracy 0.000, 19 of 21 contexts "unknown"

The climb v1 shows (0 -> 0.59 -> 0.795 in the CLI) is **memorization coverage increasing**,
not generalization — "after relearning" tests on lines it just trained on. Feed it a new
file and it collapses. Reason: "unknown" means "exact string never seen." No geometry
between words, so no connection across can ever form. That is the ceiling.

## v2 (`src/ei2.py`) — what changed and what it buys
Represents each context as a vector (hashed char 3/4-grams, self-contained, no external
model). An unseen context predicts via the **consequence of the nearest known context**
in meaning space. "Unknown" now means "nothing *near* this."

Result on the same corpus:

    [held-related]     bigram unknown 38/78  ->  v2 unknown 14/78
    [genuinely-unseen] bigram unknown  5/18  ->  v2 unknown  2/18

So v2 **cuts unknowns ~60%** — it reaches for related contexts instead of shrugging.
That is a genuine, self-contained improvement and it runs anywhere numpy runs.

## The honest limit (tested, not hand-waved)
v2's *analogy hits* are near zero (~1/24). Char-n-gram similarity finds words that
*look* alike, not words that *behave* alike ("error" and "recovery" are close in meaning,
far in spelling). Fixing that needs a **behavioural / distributional** substrate — a word
is defined by the contexts it appears in, not its characters. Prototyped that too (v3,
distributional word vectors). It also underperformed, for one decisive reason:

    the corpus is too small. 10 lines / 34 words has no redundancy to learn roles from.

This is the coverage point from the other direction: you cannot distill a meaning-space
from 10 sentences any more than learn a language from one page.

## Where "worthy" actually is
EI's architecture (learn-after-genesis, scar, persist, connect-across) is sound. The
missing ingredient is not code — it is a corpus with real redundancy. That already exists:
**149,515 messages, 496,665 entity embeddings** in the relay's store. The worthy EI runs
its predict->scar->learn loop over *that* meaning-space instead of growing its own from a
toy file. That is the merge: EI's loop + the existing embedding store = the version that can
connect across, because for the first time there is enough there to connect.

Next session, against the real store — not synthetic corpora.

## Files
- `src/ei.py`   — v1, untouched (bigram)
- `src/ei2.py`  — v2, char-embedding meaning-space (run: `python3 src/ei2.py`)
- this file      — the test results and the pointer to the real build
