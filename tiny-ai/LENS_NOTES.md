# Lens probe notes (17 Oct 2026)

Question: is the narrowing from "any word" to "the right next word" done by
running an input through the frozen model, or by how the frozen weights are
arranged?

`lens_probe.py` reads the model's own head after the embedding and after each
block, for three versions of the identical forward pass: trained, random,
and trained-then-shuffled (same numbers, arrangement destroyed).

## Result on a retrained replica of this model

The replica uses this repo's architecture and `train.py` recipe (28,256
parameters, 300 epochs). It is NOT the committed `space_llm_trained.pt`, which
was not downloaded. Run the script yourself on the committed file; numbers
will differ a little because training here is unseeded.

| weights | depth | candidates alive (of 31) | loss on true next word (bits) | top word = input word |
|---|---|---|---|---|
| trained | embed | 1.0 | 17.1 | 100% |
| trained | block 1 | 2.0 | 0.65 | 0% |
| trained | block 2 | 1.9 | 0.51 | 0% |
| random | block 2 | 28.0 | 4.96 | 64% |
| shuffled | block 2 | 5.8 | 11.5 | 20% |

Reading it:
- Random weights: no narrowing at any depth. The forward pass alone does nothing.
- Trained: the embedding row is a confident COPY of the input word (weight tying
  makes the head read the input back). Block 1 turns the copy into a prediction.
- Shuffled: it still narrows (5.8 candidates), but its loss is 11.5 bits, worse
  than guessing (4.95). Narrowing alone is not intelligence; narrowing to the
  right thing is. Elimination has to be correct elimination.
- Input dependence: the same frozen model narrows differently per prompt
  (e.g. `the dog` keeps 3 candidates, `the golden` keeps 1).

## Caveats

One model, one run. The corpus is 10 sentences and the model has memorised
it (train loss 0.27), so "prediction" here is recall of the training set.
No held-out text. The embedding-depth reading is an artifact of tied weights.
