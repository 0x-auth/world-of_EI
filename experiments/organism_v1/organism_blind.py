#!/usr/bin/env python3
"""Blind test for organism_v1: does its map predict chats it has never seen?

Learn on chats that START before --cutoff; score only chats that start on/after it (never learned from).
Vocabulary comes from the training chats only. Context resets at every chat boundary, as in organism_v1.

Models, all predicting each next message's terms from the messages before it in the same chat:
  U       message frequencies of terms in training chats (no context)
  S       snapshot: next-message co-occurrence counts from training chats, mixed with U (lambda 0.9)
  O       the forward organism, frozen at the cutoff
  O+R     forward and retro organisms averaged (retro read as effect->cause, transposed)
Score: bits per term of the next message (lower = less surprised), plus a bootstrap over test chats.

Usage (same exports as organism_v1):
  python3 organism_blind.py ~/Downloads/clean/localhost ~/Downloads/clean/phiinverse ~/Downloads/clean/merge \
          --since 2026-06-01 --cutoff 2026-10-15 --out ~/Downloads/organism_blind
"""
import argparse, json, os, numpy as np
from organism_v1 import build_chats, tokens, Organism

def ids_of(t, idx):
    return np.array(sorted(idx[w] for w in tokens(t) if w in idx), dtype=np.int64)

def update(ctx, ids, alpha):
    ctx *= alpha; ctx[ids] += (1 - alpha) / len(ids); s = ctx.sum()
    if s > 0: ctx /= s
    ctx[ctx < 1e-3] = 0

def train(chats, idx, alpha, reverse=False):
    V = len(idx); org = Organism(V); N = np.zeros((V, V), np.float32); row = np.zeros(V, np.float32)
    df = np.ones(V); n = V
    for c in chats:
        ctx = np.zeros(V, np.float32); prev = None
        for _, _, t in (c["msgs"][::-1] if reverse else c["msgs"]):
            ids = ids_of(t, idx)
            if len(ids) == 0: continue
            if ctx.any(): org.step(ctx, ids)
            if not reverse:
                df[ids] += 1; n += len(ids)
                if prev is not None: N[np.ix_(prev, ids)] += 1; row[prev] += len(ids)
            prev = ids; update(ctx, ids, alpha)
    return org, N, row, df / n

def softmax(z):
    z = z - z.max(); p = np.exp(z); return p / p.sum()

def score(chats, idx, alpha, org, orgR, N, row, U, lam=0.9):
    V = len(idx); out = {k: [] for k in ("U", "S", "O", "O+R")}; per_chat = []
    WR = orgR.W.T if orgR is not None else None
    for c in chats:
        ctx = np.zeros(V, np.float32); prev = None; bits = {k: [] for k in out}
        for _, _, t in c["msgs"]:
            ids = ids_of(t, idx)
            if len(ids) == 0: continue
            if ctx.any() and prev is not None:
                nz = np.flatnonzero(ctx); cv = ctx[nz]
                pO = softmax(org.b + cv @ org.W[nz])
                pR = softmax(orgR.b + cv @ WR[nz]) if WR is not None else pO
                m = ctx @ (N / np.maximum(row, 1)[:, None]); s = m.sum()
                pS = lam * m / s + (1 - lam) * U if s > 0 else U
                for k, p in (("U", U), ("S", pS), ("O", pO), ("O+R", 0.5 * (pO + pR))):
                    bits[k].append(float(-np.log2(np.maximum(p[ids], 1e-12)).mean()))
            prev = ids; update(ctx, ids, alpha)
        if bits["U"]:
            per_chat.append({k: float(np.mean(v)) for k, v in bits.items()} | {"n": len(bits["U"]), "title": c["title"]})
            for k in out: out[k] += bits[k]
    return out, per_chat

def boot(per_chat, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = np.array([p[a] - p[b] for p in per_chat]); w = np.array([p["n"] for p in per_chat])
    est = [np.average(d[i], weights=w[i]) for i in (rng.integers(0, len(d), len(d)) for _ in range(n))]
    return float(np.average(d, weights=w)), [float(x) for x in np.percentile(est, [2.5, 97.5])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--cutoff", required=True, help="train on chats starting before this date, test on the rest")
    ap.add_argument("--since", default=None); ap.add_argument("--human-only", action="store_true")
    ap.add_argument("--vocab", type=int, default=4000); ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--max-df", type=float, default=0.3); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--exclude", default="Cross-account chat synthesis"); ap.add_argument("--out", default="organism_blind")
    a = ap.parse_args()
    chats = build_chats(a.exports, a.human_only, a.since, [x.strip() for x in a.exclude.split(",") if x.strip()])
    tr = [c for c in chats if c["start"] < a.cutoff]; te = [c for c in chats if c["start"] >= a.cutoff]
    print(f"train {len(tr)} chats before {a.cutoff}, test {len(te)} chats after (never learned from)")
    df = {}
    for c in tr:
        for w in set().union(*(tokens(t) for _, _, t in c["msgs"])): df[w] = df.get(w, 0) + 1
    cap = a.max_df * len(tr)
    vocab = [w for w, n in sorted(df.items(), key=lambda kv: (-kv[1], kv[0])) if a.min_df <= n <= cap][:a.vocab]
    idx = {w: i for i, w in enumerate(vocab)}; print(f"vocabulary {len(vocab)} (training chats only)")
    org, N, row, U = train(tr, idx, a.alpha); orgR, _, _, _ = train(tr, idx, a.alpha, reverse=True)
    out, per = score(te, idx, a.alpha, org, orgR, N, row, U)
    res = {"train_chats": len(tr), "test_chats": len(per), "test_predictions": len(out["U"]), "bits": {k: float(np.mean(v)) for k, v in out.items()}}
    for b in ("S", "U"):
        for o in ("O", "O+R"): res[f"{o} - {b}"] = boot(per, o, b)
    os.makedirs(a.out, exist_ok=True); json.dump({"summary": res, "per_chat": per}, open(f"{a.out}/blind.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1))
    print("verdict:", "the map predicts unseen chats better than the snapshot" if res["O - S"][1][1] < 0 else "no clear gain over the snapshot")

if __name__ == "__main__":
    main()
