# EI organism v0

An organism born into a time-ordered stream (Space's own words, Jun 2024 → Aug 2025). It starts with one seed rule ("what I just said, I'll say again") and learns only from its own prediction errors. A pre-registered blind test checks whether its map predicts unseen future months better than a static snapshot.

**Result:** it does. Frozen: −0.22 bits vs the snapshot. Kept alive: −0.38 bits vs a snapshot that keeps counting. See `RESULTS.md`; the protocol is in `PREREGISTRATION.md`.

## Run
Personal data never goes in the repo; it lives in `data/` (gitignored), or wherever `DH_DIR` points.

1. Export the tables `messages`, `conversations`, `github_commits`, `github_repos` and `sources` from the D1 `digital-history` database, then load them into `data/dh.db` (SQLite).
2. `python3 build_stream.py`: builds the term stream.
3. `python3 tune.py`, then `python3 tune2.py`: validation month only.
4. `python3 test_once.py`: the blind test (Jun–Aug 2025). Writes `test_results.json` and the frozen map.
5. `python3 read_map.py`: the map's strongest links and spectral axes.
