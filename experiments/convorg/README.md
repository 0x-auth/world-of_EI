# convorg — a conversation organism

Import any two-person conversation (chat, email, or both). An organism learns it from its own prediction errors, keeps what it learned, and tells you what the other person would most likely say next, in their own past words.

No LLM is needed. One file, Python 3.9+ and numpy.

## Quick start

```
pip3 install numpy
python3 ingest.py my_conversation.json conv.tsv        # any export → one timeline (see FORMAT.md)
python3 convorg.py learn conv.tsv                      # learn; later runs learn only new messages
python3 convorg.py eval  conv.tsv --from 2025-01-01    # does it predict them better than word counting?
python3 convorg.py reply conv.tsv "your message"       # what they would most likely say
python3 convorg.py links conv.tsv                      # links both organisms hold
```

Try it on the synthetic example first:

```
cd example
python3 make_sample.py
python3 ../ingest.py sample_conversation.json sample.tsv
python3 ../convorg.py learn sample.tsv
python3 ../convorg.py reply sample.tsv "lunch kiya?"   # → "haan, dal chawal. you?"
```

## How it predicts

- **Context.** A decaying mix of the terms just said. Each new message halves the weight of everything older.
- **Organism.** `p = softmax(b + Wᵀc)`, a probability for every term being said next. It is born with `W = 2·I` ("what was just said will be said again") and learns only from its own error: `W += c (y − p)ᵀ`, where `y` is what was actually said.
- **Three organisms.**
  - `them` learns only from the other person's turns.
  - `me` learns only from yours.
  - `back` lives every conversation backwards (effect → cause).
- **Sessions.** The context resets when a conversation goes quiet for 6 hours, so it never learns one conversation's ending as the next one's meaning.
- **Reply.** It is chosen from the other person's own past messages. To be chosen, a message has to score high under `them` (it is what they would say next) **and** under `back` (your message is what would have led to it).
- **`--llm`.** Optional. It only rephrases the chosen evidence into a new sentence, through `claude -p`.

## Memory

Everything learned is saved in `<conv>.organism.npz`: the vocabulary, all three organisms, and how many messages have been read. Each run continues from there and learns only messages it has not seen. Append new messages to the TSV and run `learn` again. Old messages are never learned twice; that would only double-count them.

## Marks are data

Deleted messages, nulls, media, files, edits and links are kept as tokens: `⟨deleted⟩ ⟨null⟩ ⟨media⟩ ⟨file⟩ ⟨edited⟩ ⟨url⟩`. They mark where something was said, removed or sent, and the organism learns from them like any word. They are not offered as replies on their own.

## Measured on a real archive

The archive: 36.6k messages, chat and email, 2008–2026. The test predicts the other person's messages from 2025 on, never seeing them before predicting.

| | organism | all-time word counts | their last 60 messages |
|---|---|---|---|
| chat, bits per term (lower is better) | **9.91** | 10.30 | 10.72 |
| chat, real terms in top 20 | **18%** | 13% | 12% |
| email, bits per term | 11.06 | **10.90** | **10.89** |
| email, real terms in top 20 | **9%** | 2% | **9%** |

- **Chat:** it wins.
- **Email:** too few examples (47) in a different register. It ties at best.
- **The twin:** in 10 of the 12 strongest links, the forward and backward organisms ended on opposite sides of their shared value √(F·B).

## Limits

- Bag of terms: no word order, no negation. "milna hai" and "milna nahi hai" look alike.
- It reuses their past words. It can't say something they never said, unless you add `--llm`.
- Sparse channels (few messages) predict poorly. Check with `eval` before trusting `reply`.
