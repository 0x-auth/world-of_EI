#!/usr/bin/env python3
"""convorg: a conversation organism for any two-person conversation.

Import a conversation, let the organism learn it, then ask it what the other person would say.
Its learning is saved, so every run continues from where the last one stopped.

  pip3 install numpy
  python3 convorg.py learn  conv.tsv   # learn (first run: everything; later runs: only new messages)
  python3 convorg.py eval   conv.tsv --from 2025-01-01   # how well it predicts the other person, vs two baselines
  python3 convorg.py reply  conv.tsv "your message"  # what they would most likely say, from their own words
  python3 convorg.py reply  conv.tsv "your message" --at "2026-11-18 09:00"   # reply as if at that moment
  python3 convorg.py reply  conv.tsv "your message" --src g      # choose only from their emails
  python3 convorg.py links  conv.tsv   # strongest links both organisms hold

Input: a TSV (time<TAB>who<TAB>source<TAB>text, one message per line, oldest first, e.g. from ingest.py)
       or a JSON list of objects with a time field (ts/time/date), a speaker field (who/sender/from/direction)
       and a text field (text/body/message). --me is your speaker value (for direction-style data: "sent").
Memory: <conv>.organism.npz holds the vocabulary, three organisms and how far they have read.

How it predicts. The context c is a decaying mix of the terms just said (each message halves the weight of the
older ones). An organism maps c to a probability over every term: p = softmax(b + W^T c). It is born with W = 2·I
("what was just said will be said again") and learns only from its own error, W += c (y - p)^T, where y is
what was actually said next. Three organisms:
  them   learns only from the turns of the other person: context -> what they say next
  me     the same for you
  back   lives every conversation backwards (effect -> cause)
A reply is chosen from their own past messages: high probability under `them` for what they would say next,
AND high probability under `back` for your message being what led to it. Both directions must agree.
"""
import sys, os, re, json, math, argparse, datetime as D, collections as C
import numpy as np

STOP = set("""the and you for are was but not with this that have has had will can just its i'm what when how all from
your our out get got now also one about there then than been would could should into some any more very too only yes
okay haan hai nai nahi kya kar raha rahi rahe bhi aur gaya gayi gaye toh tha thi main mai phir tak kuch mujhe kiya
liye lena hoga hota hogi wala wali kaise kyu kyun abhi kab jab tab yeh wo woh tum hum aap mera meri mere tera teri
tumhara tumhari apna apni diya karo karna karke karte kiye dena""".split())
WORD = re.compile(r"⟨\w+⟩|[a-z][a-z0-9_']{2,}")
SESSION_GAP_H = 6
MIN_DF = 3

def toks(x):
    return {w for w in WORD.findall(x.lower()) if w not in STOP}

# ---------------- load ----------------
def load(path, me):
    rows = []
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        if isinstance(data, dict): data = next(v for v in data.values() if isinstance(v, list))
        for m in data:
            ts = str(m.get("ts") or m.get("time") or m.get("date") or "")[:16].replace("T", " ")
            who = str(m.get("who") or m.get("direction") or m.get("sender") or m.get("from") or "")
            txt = m.get("text") or m.get("body") or m.get("message") or "⟨null⟩"
            rows.append((ts, "me" if who == me else "them", m.get("source", ""), str(txt)))
        rows.sort(key=lambda r: r[0])
    else:
        for line in open(path, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4: rows.append((p[0], "me" if p[1] == me else "them", p[2], p[3]))
    out = []
    for ts, w, s, x in rows:
        try: t = D.datetime.strptime(ts[:16], "%Y-%m-%d %H:%M")
        except ValueError: continue
        out.append((t, w, s, x))
    return out

# ---------------- organism ----------------
class Org:
    def __init__(self, V, W=None, b=None):
        if W is None:
            W = np.zeros((V, V), np.float32); np.fill_diagonal(W, 2.0); b = np.zeros(V, np.float32)
        self.W, self.b = W.astype(np.float32), b.astype(np.float32)
    def grow(self, V):
        n = self.W.shape[0]
        if V <= n: return
        W = np.zeros((V, V), np.float32); W[:n, :n] = self.W; W[np.arange(n, V), np.arange(n, V)] = 2.0
        self.W = W; self.b = np.concatenate([self.b, np.zeros(V - n, np.float32)])
    def predict(self, ctx):
        nz = np.flatnonzero(ctx); z = self.b + ctx[nz] @ self.W[nz]; z -= z.max(); p = np.exp(z); return p / p.sum()
    def learn(self, ctx, ids, p=None):
        p = self.predict(ctx) if p is None else p
        y = np.zeros_like(p); y[ids] = 1 / len(ids); g = (y - p).astype(np.float32)
        nz = np.flatnonzero(ctx); self.W[nz] += np.outer(ctx[nz], g); self.b += g
        return p

def push(ctx, ids, alpha=0.5):
    ctx *= alpha; ctx[ids] += (1 - alpha) / len(ids); ctx /= ctx.sum(); ctx[ctx < 1e-3] = 0

def sessions(rows, start=0):
    S, cur = [], None
    for i in range(start, len(rows)):
        r = rows[i]
        if cur and (r[0] - rows[cur[-1]][0]).total_seconds() < SESSION_GAP_H * 3600: cur.append(i)
        else: cur = [i]; S.append(cur)
    return S

class Brain:
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            self.vocab = list(z["vocab"]); self.read = int(z["read"]); self.runs = int(z["runs"])
            V = len(self.vocab)
            self.o = {k: Org(V, z[f"W_{k}"].astype(np.float32), z[f"b_{k}"]) for k in ("them", "me", "back")}
        else:
            self.vocab, self.read, self.runs, self.o = [], 0, 0, {}
        self.idx = {w: i for i, w in enumerate(self.vocab)}
    def save(self):
        np.savez_compressed(self.path, vocab=np.array(self.vocab), read=self.read, runs=self.runs,
                            **{f"W_{k}": o.W.astype(np.float16) for k, o in self.o.items()},
                            **{f"b_{k}": o.b for k, o in self.o.items()})
    def extend_vocab(self, rows, cap, max_df=None):
        df = C.Counter(); S = sessions(rows)
        for s in S:
            df.update(set().union(*(toks(rows[i][3]) for i in s)))
        # leave out terms present in almost every conversation (they predict nothing).
        # big archives: 15% of sessions; small ones need a looser cut or nothing is left
        max_df = max_df or (0.15 if len(S) > 1000 else 0.8)
        maxdf = max_df * max(len(S), 1)
        new = [w for w, n in sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))
               if MIN_DF <= n <= maxdf and w not in self.idx][:max(cap - len(self.vocab), 0)]
        self.vocab += new; self.idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        for k in ("them", "me", "back"):
            if k in self.o: self.o[k].grow(V)
            else: self.o[k] = Org(V)
        return len(new)
    def ids(self, x):
        return np.array(sorted(self.idx[w] for w in toks(x) if w in self.idx), dtype=np.int64)

def live(brain, rows, start, end=None, on_predict=None):
    """forward through rows[start:end] (new messages only), then the backward twin over the same sessions"""
    V = len(brain.vocab); end = len(rows) if end is None else end
    S = [s for s in sessions(rows[:end], 0) if s[-1] >= start]
    for s in S:
        ctx = np.zeros(V, np.float32)
        for i in s:
            ids = brain.ids(rows[i][3])
            if len(ids) == 0: continue
            if ctx.any() and i >= start:
                who = rows[i][1]
                p = brain.o[who].predict(ctx)
                if on_predict: on_predict(i, ids, p, ctx)
                brain.o[who].learn(ctx, ids, p)
            push(ctx, ids)
    for s in S:
        ctx = np.zeros(V, np.float32)
        for i in s[::-1]:
            ids = brain.ids(rows[i][3])
            if len(ids) == 0: continue
            if ctx.any() and i >= start: brain.o["back"].learn(ctx, ids)
            push(ctx, ids)

def context_at(brain, rows, upto, n=12):
    ctx = np.zeros(len(brain.vocab), np.float32)
    for t, w, s, x in rows[max(0, upto - n):upto]:
        ids = brain.ids(x)
        if len(ids): push(ctx, ids)
    return ctx

# ---------------- commands ----------------
def cmd_learn(a, rows):
    br = Brain(a.state)
    added = br.extend_vocab(rows, a.vocab, a.max_df)
    new = len(rows) - br.read
    if new <= 0 and not added:
        print(f"nothing new: {br.read} messages already learned over {br.runs} run(s)"); return br
    live(br, rows, br.read)
    br.read = len(rows); br.runs += 1; br.save()
    print(f"run {br.runs}: learned {max(new,0)} new messages (total {br.read}), vocabulary {len(br.vocab)} (+{added}) -> {a.state}")
    return br

def cmd_eval(a, rows):
    cut = D.datetime.strptime(a.__dict__["from"], "%Y-%m-%d")
    k = next((i for i, r in enumerate(rows) if r[0] >= cut), len(rows))
    br = Brain(a.state + ".eval.tmp"); br.extend_vocab(rows[:k], a.vocab, a.max_df)
    live(br, rows, 0, k)
    V = len(br.vocab); uni = np.ones(V); recent = C.deque(maxlen=60)
    for r in rows[:k]:
        ids = br.ids(r[3])
        if len(ids): uni[ids] += 1
        if len(ids) and r[1] == "them": recent.append(ids)
    res = C.defaultdict(list)
    def onp(i, ids, p, ctx):
        if rows[i][1] != "them": return
        rc = np.full(V, 0.1)
        for q in recent: rc[q] += 1
        for name, q in (("organism", p), ("all-time words", uni / uni.sum()), ("their last 60", rc / rc.sum())):
            res[(rows[i][2] or "all", name)].append((float(-np.log2(np.maximum(q[ids], 1e-12)).mean()),
                                                      len(set(np.argsort(-q)[:20]) & set(ids)) / len(ids)))
    def onp2(i, ids, p, ctx):
        onp(i, ids, p, ctx); uni[ids] += 1
        if rows[i][1] == "them": recent.append(ids)
    live(br, rows, k, None, onp2)
    print(f"predicting the other person's messages from {a.__dict__['from']} on (learning as it goes, vocabulary frozen at the cut)")
    for src in sorted({s for s, _ in res}):
        line = [f"{src or 'all':>6}:"]
        for name in ("organism", "all-time words", "their last 60"):
            r = np.array(res[(src, name)]); line.append(f"{name} {r[:,0].mean():.2f} bits, top-20 {r[:,1].mean():.0%}")
        print(f"  n={len(res[(src,'organism')]):5} " + " | ".join(line))
    print("  lower bits = better prediction; top-20 = share of their real terms found in the 20 most likely")

def cmd_reply(a, rows):
    br = Brain(a.state)
    if not br.vocab: print("run learn first"); return
    upto = len(rows)
    if a.at:
        at = D.datetime.strptime(a.at, "%Y-%m-%d %H:%M"); upto = next((i for i, r in enumerate(rows) if r[0] >= at), len(rows))
        if upto < br.read: print(f"(note: the saved organism has read up to message {br.read}; for a clean test use eval)")
    ctx = context_at(br, rows, upto); mine = br.ids(a.text)
    if len(mine): push(ctx, mine)
    p = br.o["them"].predict(ctx); logp = np.log(np.maximum(p, 1e-12))
    ref = rows[upto - 1][0] if upto else D.datetime.now()
    cands = []
    for i in range(upto):
        if rows[i][1] != "them" or (a.src and rows[i][2] != a.src): continue
        ids = br.ids(rows[i][3])
        if len(ids) < 1 or all(br.vocab[j].startswith("⟨") for j in ids): continue   # marks shape predictions, but a reply needs words
        age = max((ref - rows[i][0]).days, 0)
        cands.append((logp[ids].mean() + a.recency * math.log(0.5 ** (age / 365) + 1e-3), i, ids))
    cands.sort(key=lambda z: -z[0]); cands = cands[:200]
    scored = []
    for s1, i, ids in cands:                         # the twin: does their message lead back to yours?
        c2 = np.zeros(len(br.vocab), np.float32); push(c2, ids)
        pb = br.o["back"].predict(c2)
        s2 = float(np.log(np.maximum(pb[mine], 1e-12)).mean()) if len(mine) else 0.0
        scored.append((s1 + s2, s1, s2, i))
    scored.sort(key=lambda z: -z[0])
    top = [br.vocab[j] for j in np.argsort(-p)[:25]]
    print("predicted terms:", " ".join(top))
    print("\nmost likely replies, in their own words (forward + backward agree):")
    for tot, s1, s2, i in scored[:a.k]:
        print(f"  [{rows[i][0]:%Y-%m-%d}] {rows[i][3][:400]}")
    if a.llm:
        import subprocess
        prompt = ("Write the reply the other person would most likely send now, in their style for this channel. "
                  "Use these predicted terms and these past replies of theirs as evidence; do not invent facts.\n\n"
                  f"Recent conversation:\n" + "\n".join(f"{r[1]}: {r[3][:500]}" for r in rows[max(0, upto-10):upto]) +
                  f"\nme: {a.text}\n\nPredicted terms: {' '.join(top)}\nTheir closest past replies:\n" +
                  "\n".join(f"- [{rows[i][0]:%Y-%m-%d}] {rows[i][3][:400]}" for _, _, _, i in scored[:a.k]))
        r = subprocess.run(["claude", "-p"], input=prompt, capture_output=True, text=True, timeout=300)
        print("\nLLM phrasing (labelled prediction):\n", r.stdout.strip() or r.stderr[-300:])

def cmd_links(a, rows):
    br = Brain(a.state)
    off = lambda W: np.clip(W - np.diag(np.diag(W)), 0, None)
    F = (off(br.o["them"].W) + off(br.o["me"].W)) / 2; L = np.sqrt(F * off(br.o["back"].W).T); np.fill_diagonal(L, 0)
    for k in np.argsort(-L, axis=None)[:a.k * 6]:
        u, v = np.unravel_index(k, L.shape); print(f"  {br.vocab[u]} -> {br.vocab[v]}   {L[u, v]:.2f}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["learn", "eval", "reply", "links"])
    ap.add_argument("conv"); ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--me", default="me", help="your speaker value in the file (default 'me', as written by ingest.py)")
    ap.add_argument("--state", default=None, help="memory file (default: <conv>.organism.npz)")
    ap.add_argument("--vocab", type=int, default=3000)
    ap.add_argument("--max-df", type=float, default=None, help="drop terms found in more than this share of sessions (default: 0.15 for big archives, 0.8 for small)")
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--from", default="2025-01-01"); ap.add_argument("--at", default="")
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--recency", type=float, default=0.5)
    ap.add_argument("--src", default="", help="only choose replies from this source tag (e.g. g for email, w for WhatsApp)")
    ap.add_argument("--llm", action="store_true", help="also phrase a new reply with claude -p")
    a = ap.parse_args(); a.state = a.state or a.conv + ".organism.npz"
    global MIN_DF; MIN_DF = a.min_df
    rows = load(a.conv, a.me)
    print(f"{len(rows)} messages, {rows[0][0]:%Y-%m-%d} .. {rows[-1][0]:%Y-%m-%d}")
    {"learn": cmd_learn, "eval": cmd_eval, "reply": cmd_reply, "links": cmd_links}[a.cmd](a, rows)

if __name__ == "__main__":
    main()
