#!/usr/bin/env python3
"""
prime_walk.py — walk the primes in their one direction, as far as you like.

Segmented sieve: memory stays small (one segment at a time), so the limit is only
time. State is saved after every segment, so you can kill it at any moment and
run it again: it resumes where it stopped (the seed's "survives process death").

What it measures as it walks:
  1. last digit of each prime -> last digit of the next one (transition table)
  2. correlation between consecutive gaps
  3. too-high / too-low: after a gap above its expected size (log p), how often
     is the next gap also above?

Usage:
  python3 prime_walk.py --primes 100000000          # walk until 100M primes
  python3 prime_walk.py --limit 5000000000          # or walk up to a number
  python3 prime_walk.py --report                    # print results so far
  python3 prime_walk.py --reset                     # start over

Needs: numpy.  Speed here: 100M primes in ~40 s; 1B primes is roughly 10x that.
"""
import argparse, json, math, os, sys, time
import numpy as np

STATE = os.path.expanduser("~/.prime_walk_state.json")
SEG = 1 << 24          # odd numbers per segment (~16.7M, ~16 MB of memory)
DIGS = [1, 3, 7, 9]


def base_primes(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return np.nonzero(s)[0]


def new_state():
    return {"next_odd": 11, "count": 4, "last_p": 7, "last_gap": None,
            "trans": [[0] * 4 for _ in range(4)],
            "g_n": 0, "g_sum": 0.0, "g_sq": 0.0, "g_cross": 0.0,   # gap statistics
            "hi_after_hi": 0, "hi": 0, "hi_after_lo": 0, "lo": 0, "seconds": 0.0}


def load():
    if os.path.exists(STATE):
        with open(STATE) as f: return json.load(f)
    return new_state()


def save(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f: json.dump(st, f)
    os.replace(tmp, STATE)                      # atomic: a kill never corrupts it


def walk_segment(st, bp):
    lo = st["next_odd"]                         # odd start
    hi = lo + 2 * SEG                           # exclusive
    seg = np.ones(SEG, dtype=bool)              # seg[i] <-> lo + 2i
    for q in bp[1:]:                            # skip 2
        if q * q >= hi: break
        start = max(q * q, ((lo + q - 1) // q) * q)
        if start % 2 == 0: start += q
        seg[(start - lo) // 2::q] = False
    ps = lo + 2 * np.nonzero(seg)[0]
    st["next_odd"] = hi
    if len(ps) == 0: return ps
    seq = np.concatenate(([st["last_p"]], ps)).astype(np.int64)
    lut = np.full(10, -1, dtype=np.int64); lut[[1, 3, 7, 9]] = [0, 1, 2, 3]
    d = lut[seq % 10]
    t = np.bincount(d[:-1] * 4 + d[1:], minlength=16).reshape(4, 4)
    st["trans"] = (np.array(st["trans"]) + t).tolist()
    g = np.diff(seq).astype(np.float64)
    if st["last_gap"] is not None: g = np.concatenate(([st["last_gap"]], g))
    st["g_n"] += len(g) - (1 if st["last_gap"] is not None else 0)
    st["g_sum"] += float(g[1:].sum() if st["last_gap"] is not None else g.sum())
    st["g_sq"] += float((g[1:] ** 2).sum() if st["last_gap"] is not None else (g ** 2).sum())
    st["g_cross"] += float((g[:-1] * g[1:]).sum())
    # too-high / too-low against expected gap log p
    pp = seq[1:] if st["last_gap"] is None else np.concatenate(([st["last_p"]], seq[1:]))
    above = g > np.log(pp.astype(np.float64))
    prev, nxt = above[:-1], above[1:]
    st["hi"] += int(prev.sum()); st["hi_after_hi"] += int((prev & nxt).sum())
    st["lo"] += int((~prev).sum()); st["hi_after_lo"] += int((~prev & nxt).sum())
    st["last_gap"] = float(g[-1]); st["last_p"] = int(seq[-1]); st["count"] += len(ps)
    return ps


def report(st):
    T = np.array(st["trans"], dtype=float)
    if T.sum() == 0:
        print("nothing walked yet: run with --primes N or --limit N"); return
    print(f"\nprimes walked: {st['count']:,}   up to {st['last_p']:,}   time {st['seconds']/60:.1f} min")
    print("\n1) last digit -> next prime's last digit (row %; no memory would be 25 each)")
    print("        " + "".join(f"{x:>8}" for x in DIGS))
    for i, x in enumerate(DIGS):
        print(f"   {x}   " + "".join(f"{v:>7.2f}%" for v in T[i] / T[i].sum() * 100))
    n = st["g_n"]
    if n > 2:
        m = st["g_sum"] / n; v = st["g_sq"] / n - m * m
        r = (st["g_cross"] / (n - 1) - m * m) / v
        print(f"\n2) correlation of consecutive gaps: {r:+.5f}   (noise level ~ {1/math.sqrt(n):.5f})")
    if st["hi"] and st["lo"]:
        print(f"3) next gap above expected: {st['hi_after_hi']/st['hi']*100:.2f}% after an above gap, "
              f"{st['hi_after_lo']/st['lo']*100:.2f}% after a below gap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int); ap.add_argument("--limit", type=int)
    ap.add_argument("--report", action="store_true"); ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.reset and os.path.exists(STATE): os.remove(STATE)
    st = load()
    if a.report or not (a.primes or a.limit):
        report(st); return
    target_n = a.limit or 10 ** 18
    bp = base_primes(int(math.isqrt(st["next_odd"] + 4 * SEG)) * 2)   # grows as the walk goes
    t0 = time.time()
    try:
        while (a.primes is None or st["count"] < a.primes) and st["next_odd"] < target_n:
            if bp[-1] ** 2 < st["next_odd"] + 2 * SEG:          # extend base primes if needed
                bp = base_primes(int(math.isqrt(st["next_odd"] + 4 * SEG)) * 2)
            ts = time.time(); walk_segment(st, bp); st["seconds"] += time.time() - ts
            save(st)
            print(f"\r  {st['count']:>14,} primes, up to {st['last_p']:>16,}   "
                  f"({(time.time()-t0)/60:5.1f} min this run)", end="", flush=True)
    except KeyboardInterrupt:
        print("\n  stopped; state saved, run again to resume")
    report(st)


if __name__ == "__main__":
    main()
