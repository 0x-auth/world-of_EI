# EI organism

A frozen rule that learns only from its own prediction error, lives once over your threads,
and passes a corrected seed to the next life. It finds **hidden links**: pairs of threads that are
strongly tied through what the organism learned, yet share few words.

```
Seed ──► Medium ──► Rule ──► Filter ──► Seed
carried   any DB     softmax(b + Wᵀc)   strikes,      corrected weights
weights   as events  forward + retro    df / hub      handed on
```

| Part   | What it is | File |
|--------|-----------|------|
| Seed   | both twins' weights + vocabulary, saved each run; struck pairs are cut out before it is carried | `organism/seed.py` |
| Medium | any source that shows `events(thread_id, time, role, text)` (+ optional `threads(thread_id, title, account, updated)`) | `organism/sources.py` |
| Rule   | `softmax(b + Wᵀc)`, `dW = η·c(y−p)ᵀ`, born `W = 2I`; forward and retrocausal twin; link = √(F·Bᵀ) | `organism/core.py` |
| Filter | min/max document frequency, idf, hub lift, strikes read back from the store | `organism/core.py`, `organism/store.py` |

The maths is organism v1.4 unchanged; with `--parts 0`, `tests/test_framework.py` checks the links match v1.4 exactly.

**Parts (default).** A long thread touches everything, so as one unit it links to nothing in particular. By default
links are made between parts of 20 messages (`--parts 20`); each thread pair keeps its strongest part pair, and each
link says where it lives (`msgs 41–60`). On 50 real chats this took the largest research chats from zero links to the
most linked. `--parts 0` gives whole-thread links as in v1.4.

**Known limit.** It links through learned word pairs. An idea that returns in different words is missed.

## Use

```bash
pip install .
organism clean ~/Downloads/06-11-2026-claude-exports ~/Downloads/clean       # Claude exports only
organism run ~/Downloads/clean/localhost ~/Downloads/clean/phiinverse ~/Downloads/clean/merge --since 2026-08-01
organism strike <run_id> 3 --reason "different meaning"     # wrong link: its term pairs never carry a link again
organism keep   <run_id> 1                                  # right link
organism status
organism run ...                                            # next life: starts from the last seed (--seed none = newborn)
```

## Any database

Make a view with the four columns and point the organism at it:

```sql
CREATE VIEW events  AS SELECT conversation_id AS thread_id, created_at AS time, sender AS role, body AS text FROM messages;
CREATE VIEW threads AS SELECT id AS thread_id, name AS title, owner AS account, updated_at AS updated FROM conversations;
```

| Source | Spec |
|--------|------|
| export folder | `path/to/folder` |
| SQLite | `sqlite:file.db` |
| Postgres | `postgres:postgresql://...` (`pip install .[postgres]`) |
| Cloudflare D1 | `d1:<database-uuid>`, with `CF_ACCOUNT_ID` and `CF_API_TOKEN` in the environment |

The organism's own memory (`organism_runs`, `organism_links`, `organism_strikes`) goes to `--store`:
a SQLite path (default `organism_out/organism.db`), `d1:<uuid>` or `postgres:DSN`.

## On Cloudflare

`cloudflare/` runs it next to the data: a Worker and Durable Object feed a container from D1, keep the seed in R2,
and write links back to D1. See `cloudflare/README.md`. `organism/server.py` is the container's HTTP face.

## Docker

The image holds code only. Data is mounted read-only at run time and never baked in.

```bash
docker build -t ei-organism:0.1.0 .
docker run --rm -v ~/Downloads/clean:/data:ro -v "$PWD":/work ei-organism:0.1.0 \
  run /data/localhost /data/phiinverse /data/merge --since 2026-08-01
```

## Tests

```bash
pip install .[test] && pytest -q
```
Synthetic threads only (`tests/make_fixture.py`); no real chats are in this repository.
