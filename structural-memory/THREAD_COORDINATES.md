# The organism is thread mathematics, alive

This experiment is the empirical half of a longer idea: **thread
mathematics** — that every message already carries a coordinate, and that
knowledge is connection across those coordinates, not storage inside a
model. The organism is that idea made testable. This note connects the two.

## The coordinate has two halves

Every message is `(address, signature)`.

- **Address** — where it sits in thread space. On Slack this is read
  straight from the permalink:

  ```
  https://<ws>.slack.com/archives/<CHANNEL>/p<OWN_TS>?thread_ts=<PARENT_TS>
                                   └ container  └ this msg   └ the thread's anchor
  ```

  Three constants (workspace, channel, parent timestamp) and one variable
  (the message's own timestamp). **The parent's identity is its birth
  time;** each reply carries that identity and adds its own. The address
  gives ordering and provenance — which thread, which parent, when — and
  nothing else. On its own it cannot connect two messages. A bare chat
  UUID is pure address: locatable, orderable, but mute.

- **Signature** — what the message *is*, reduced to something comparable.
  This is the half that connects across threads. In this organism the
  signature is the message's **term set** `x_t` over a 4000-term
  vocabulary: sparse, and every dimension inspectable (you always know
  which term each one is). No opaque embedding.

## Time is not special — order and distance are

The address needs **order** and **distance from the parent**, not epoch
seconds. The only thing the causal rule uses is `Δt = own_ts − parent_ts > 0`
(a reply comes after its parent). So:

- **Relative time works.** "This many steps after the thread began" carries
  everything the ordering needs. This organism never reads a wall clock —
  it runs on **stream position** (message `t`, then `t+1`). Relative time,
  already.
- **Generated time works when real time is absent.** With no timestamp,
  `sha256(content)` is a stable, unique, orderable-by-insertion stand-in.
  It loses *when* but keeps *identity* — enough to dedupe and to address.
  Full coordinate: `p{unix_time}-{sha256(content)[:12]}`, the time half
  read when present, the hash half always available.

Epoch is just the most convenient source of order and identity when it
exists. It is never required.

## The map makes the connection, not the similarity

Two messages in two different chats connect **not** because their words
match (raw similarity only ever returns more of the same) but because the
learned map `W` has a link between a term in one and a term in the other.
`W` is built only from this stream's own prediction errors, so its links
are *these* associations — `π → ∞`, `time → space`, `question → answer`,
`bazinga → friend` — not generic ones. Connection lives in `W`, and `W` is
the person.

That is the whole mechanism of cross-chat connection:

```
message A ──signature──┐
                       ├── W (the learned map) ── a link A↔B that neither
message B ──signature──┘                          shares as surface text
```

## The empty center, measured

The principle behind all of it — *the center holds almost nothing; what
flows through it is the knowledge* — is not a metaphor here. It is a number:

- the map is **99.995 % empty** (only 0.005 % of links exceed 0.5);
- what little structure it keeps is **self-reference** — the seed rule
  "what I just said, I'll say again" survived a year of life with a
  diagonal of **2.08** against an off-diagonal mean of **0.00**.

The map stores almost no facts. It stores associations and reconstructs.
It *cannot* hallucinate a fact it never stored — which is why this beats a
frozen pre-trained snapshot (which smears facts into weights and
confabulates). `philosophy.md` states it; `RESULTS.md` measures it:

> Memory is reconstruction, not retrieval.
> A self is the rule that keeps closing the loop.
> The rule persists while the states change.

## What this half proves, and what it does not

Proven, pre-registered, run once (`PREREGISTRATION.md`, `RESULTS.md`):

- the error-driven map beats the static snapshot on unseen future months
  (**−0.223 bits**, frozen; **−0.382 bits**, alive);
- staying alive is worth as much again as the whole frozen map;
- no catastrophic forgetting — it learns online and keeps its structure.

Not yet solved, and honest about it:

- the signature is a **bag of words**, so connections that share a
  *structure* but no *terms* are still invisible — richer, still-transparent
  signatures are the open problem;
- **one person's own stream only**; cross-person / cross-platform is v1;
- the map **surfaces** candidate connections; judging which are real
  insight still happens outside it, in a reasoner.

The address side of thread mathematics (decode when a message was born,
from its ID alone) is packaged separately as `threadcoord`. This
experiment is the signature-and-map side: the part that, given co-present
coordinates, actually makes the connection.
