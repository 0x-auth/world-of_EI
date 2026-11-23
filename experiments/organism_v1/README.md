# organism v1.4

Next to [organism_v0](../organism_v0). `softmax(b + Wᵀc)`, W₀ = 2I, ΔW = η·c(y−p)ᵀ. It has a forward organism and a retro one, links = sqrt(F·Wbᵀ), a hub correction, idf, and a context reset per chat. v1.4 replaces v1.1.

- `clean_exports.py` cleans the Claude exports. The data stays local (see `.gitignore`).
- `organism_v1.py` builds the chats, trains, and writes the links, the map and the SQL for digital-history.
- `organism_blind.py` is the held-out test. It learns on chats before `--cutoff` and scores the chats after it, in bits per term, against U (frequencies) and S (snapshot co-occurrence). O has to beat S for the links to be a map rather than word overlap.

```
python3 organism_blind.py --cutoff 2026-09-01 --out blind_out
python3 organism_blind.py --cutoff 2026-09-01 --human-only --out blind_out_h
```

Synthetic check: O − S is −0.48 bits when message order carries structure and +0.14 when the order is shuffled, so the test can fail.
