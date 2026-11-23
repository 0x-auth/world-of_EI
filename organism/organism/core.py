"""The rule. Same maths as organism_v1.py v1.4; only the input shape changed (threads of events)."""
import re, hashlib
import numpy as np

STOP = set("""a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing don down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off
on once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which while
who whom why will with would you your yours yourself yourselves also like one get got let lets yeah okay ok yes right
know think mean really actually basically something thing things want need use using used make made way see say said
im i'm it's that's don't can't you're let's i've i'll""".split()) | set("""claude mr hey hehe haha bro dude man yaar bhai
pls please plz thanks thank sorry wait cos coz sud shud wud cud ur tell told ask asked still sure fine good great nice
cool awesome done okay alright hmm umm uh um gonna wanna kinda lol yep nope""".split())
WORD = re.compile(r"[a-z][a-z0-9_'\-]{2,}|[φπΔ∆∞ψΩτθ√]")
MATH = re.compile(r"[Δ∆]\s?[a-zA-Zτ]|√\s?\d+|\b[a-z]\s*[+\-−]\s*i?\s*1?\s*/\s*[a-z]\b|[\-−]\s*1\s*/\s*[a-z]\b"
                  r"|\bi\s*/\s*[a-z]\b|\b[a-z]{2}\s*=\s*[\-−]?\s*1\b|\b\d{1,2}\s*/\s*\d{1,2}\b")
MAXCH = 4000


def tokens(text):
    math = {re.sub(r"\s+", "", x).replace("−", "-").replace("∆", "Δ") for x in MATH.findall(text)}
    return {w for w in WORD.findall(text.lower()) if w not in STOP} | math


# ---------------- threads from events ----------------
def _ts(t):
    import datetime
    try:
        d = datetime.datetime.fromisoformat(str(t).replace("Z", "+00:00").replace(" ", "T", 1))
        return (d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)).timestamp()
    except ValueError:
        return None


def split_gaps(chats, hours):
    """One long-running thread (a WhatsApp chat of years) becomes sessions: cut wherever the gap between
    two messages is over `hours`. A thread that splits gets ids id#0, id#1 ... and its later sessions
    carry their first date in the title. Threads that don't split are unchanged."""
    out = []
    for c in chats:
        sessions, cur, last = [], [], None
        for m in c["msgs"]:
            ts = _ts(m[0])
            if cur and last is not None and ts is not None and ts - last > hours * 3600:
                sessions.append(cur); cur = []
            cur.append(m)
            if ts is not None: last = ts
        if cur: sessions.append(cur)
        if len(sessions) == 1:
            out.append(c); continue
        for k, ms in enumerate(sessions):
            out.append({**c, "id": f"{c['id']}#{k}", "msgs": ms, "start": ms[0][0],
                        "title": c["title"] if k == 0 else f"{c['title']} · {ms[0][0][:10]}"})
    out.sort(key=lambda c: c["start"])
    return out


def threads_from_events(events, meta=None, human_only=False, since=None, exclude=()):
    """events: iterable of (thread_id, time, role, text). meta: {thread_id: {title, account, updated}}.
    Returns threads sorted by start, each {id, acct, title, start, updated, msgs:[(time, role, text)]}."""
    meta = meta or {}
    by = {}
    for tid, t, role, text in events:
        by.setdefault(tid, []).append((t or "", role, text or ""))
    out, seen = [], set()
    for tid in by:   # source order kept, as in v1.4 (dedup keeps the first copy)
        m = meta.get(tid, {})
        title = m.get("title") or "Untitled"
        upd = m.get("updated") or max(x[0] for x in by[tid])
        if any(x.lower() in title.lower() for x in exclude) or (since and upd < since):
            continue
        msgs = []
        for t, role, text in sorted(by[tid], key=lambda x: x[0]):
            if human_only and role not in ("human", "user"):
                continue
            text = text[:MAXCH]
            if not text.strip():
                continue
            h = hashlib.sha1(text.encode()).hexdigest()
            if len(text) >= 40 and h in seen:
                continue
            seen.add(h)
            msgs.append((t, role, text))
        if msgs:
            out.append({"id": tid, "acct": m.get("account") or "", "title": title, "start": msgs[0][0],
                        "updated": upd, "msgs": msgs})
    out.sort(key=lambda c: c["start"])
    return out


def build_vocab(chats, size=4000, min_df=3, max_df=0.3):
    df = {}
    for c in chats:
        for w in set().union(*(tokens(t) for _, _, t in c["msgs"])):
            df[w] = df.get(w, 0) + 1
    cap = max_df * len(chats)
    common = sorted(w for w, n in df.items() if n > cap)
    vocab = [w for w, n in sorted(df.items(), key=lambda kv: (-kv[1], kv[0])) if min_df <= n <= cap][:size]
    return vocab, common


# ---------------- the organism ----------------
class Organism:
    """softmax(b + W^T c). Born with W = seed*I, or with a carried seed. Learns only from its own error."""
    def __init__(self, V, eta=1.0, seed=2.0, W=None, b=None):
        if W is None:
            self.W = np.zeros((V, V), np.float32); np.fill_diagonal(self.W, seed)
        else:
            self.W = W.astype(np.float32, copy=True)
        self.b = np.zeros(V, np.float32) if b is None else b.astype(np.float32, copy=True)
        self.eta = eta

    def step(self, ctx, ids):
        """The one hot function. A Rust kernel can replace exactly this."""
        nz = np.flatnonzero(ctx); cv = ctx[nz]
        z = self.b + cv @ self.W[nz]; z -= z.max(); p = np.exp(z); p /= p.sum()
        bits = float(-np.log2(np.maximum(p[ids], 1e-12)).mean())
        y = np.zeros_like(p); y[ids] = 1.0 / len(ids)
        g = (self.eta * (y - p)).astype(np.float32)
        self.W[nz] += np.outer(cv, g); self.b += g
        return bits


def live(chats, vocab, alpha=0.5, reverse=False, born=None):
    """reverse=True: the retrocausal twin lives each chat from its last message to its first.
    born: (W, b) carried seed already aligned to vocab, or None for a newborn."""
    idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    org = Organism(V) if born is None else Organism(V, W=born[0], b=born[1])
    bits, n = [], 0
    for c in chats:
        ctx = np.zeros(V, np.float32)
        for _, _, t in (c["msgs"][::-1] if reverse else c["msgs"]):
            ids = np.array(sorted(idx[w] for w in tokens(t) if w in idx), dtype=np.int64)
            if len(ids) == 0:
                continue
            if ctx.any():
                bits.append(org.step(ctx, ids)); n += 1
            ctx *= alpha; ctx[ids] += (1 - alpha) / len(ids); ctx /= ctx.sum(); ctx[ctx < 1e-3] = 0
    k = min(2000, max(1, n // 4))
    return org, n, (float(np.mean(bits[:k])) if bits else 0.0), (float(np.mean(bits[-k:])) if bits else 0.0)


def link_matrix(Wf, Wb=None):
    off = lambda W: np.clip(W - np.diag(np.diag(W)), 0, None)
    F = off(Wf)
    L = F if Wb is None else np.sqrt(F * off(Wb).T)
    return (L + L.T) / 2


def apply_strikes(L, vocab, struck_pairs):
    """Filter: a struck term pair carries no link, in either direction."""
    idx = {w: i for i, w in enumerate(vocab)}
    n = 0
    for u, v in struck_pairs:
        if u in idx and v in idx:
            L[idx[u], idx[v]] = L[idx[v], idx[u]] = 0; n += 1
    return n


def chat_links(chats, vocab, L, top=60, focus=None, min_msgs=6):
    idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    X = np.zeros((len(chats), V))
    for k, c in enumerate(chats):
        for _, _, t in c["msgs"]:
            for w in tokens(t):
                if w in idx: X[k, idx[w]] = 1
    idf = np.log((1 + len(chats)) / (1 + X.sum(0))) + 1
    Xw = X * idf
    Xn = Xw / np.maximum(Xw.sum(1, keepdims=True), 1e-9)
    surface = Xw / np.maximum(np.linalg.norm(Xw, axis=1, keepdims=True), 1e-9)
    surface = surface @ surface.T
    mapped = Xn @ L @ Xn.T
    np.fill_diagonal(mapped, 0)
    row = mapped.sum(1) / max(len(chats) - 1, 1)
    mapped = mapped / np.maximum(np.outer(row, row) / max(row.mean(), 1e-12), 1e-12)
    np.fill_diagonal(mapped, 0)
    iu = np.triu_indices(len(chats), 1)
    big = np.array([len(c["msgs"]) >= min_msgs for c in chats])
    keep = big[iu[0]] & big[iu[1]]
    iu = (iu[0][keep], iu[1][keep])
    if focus is not None:
        keep = np.array([a in focus and b in focus for a, b in zip(*iu)], dtype=bool)
        iu = (iu[0][keep], iu[1][keep])
    if len(iu[0]) == 0:
        return []
    m, s = mapped[iu], surface[iu]
    hidden = (m / max(m.max(), 1e-12)) * (1 - s)
    order = np.argsort(-hidden, kind="stable")[:top]
    out = []
    for o in order:
        a, b = iu[0][o], iu[1][o]
        ta, tb = np.flatnonzero(X[a]), np.flatnonzero(X[b])
        sub = L[np.ix_(ta, tb)]
        carry = [(vocab[ta[i]], vocab[tb[j]], float(sub[i, j]))
                 for i, j in zip(*np.unravel_index(np.argsort(-sub, axis=None)[:5], sub.shape)) if sub[i, j] > 0]
        out.append({"a": chats[a]["id"], "a_title": chats[a]["title"], "a_acct": chats[a]["acct"],
                    "b": chats[b]["id"], "b_title": chats[b]["title"], "b_acct": chats[b]["acct"],
                    "map": float(m[o]), "words": float(s[o]), "hidden": float(hidden[o]),
                    "cross_account": chats[a]["acct"] != chats[b]["acct"], "carried_by": carry})
    return out


def part_links(chats, vocab, L, size=20, top=60, focus=None, min_msgs=6, block=1024):
    """Link parts of threads (size messages each) instead of whole threads, so a long thread links
    through the part that carries the idea. Sessions of one thread never link to each other; each pair of
    threads keeps its strongest part pair. Score: map strength against both parts' usual links (lift).
    Works in row blocks, so memory grows with parts x vocabulary, not parts squared."""
    idx = {w: i for i, w in enumerate(vocab)}
    keep = [i for i, c in enumerate(chats) if len(c["msgs"]) >= min_msgs and (focus is None or i in focus)]
    parts = [(i, s, min(s + size, len(chats[i]["msgs"]))) for i in keep
             for s in range(0, len(chats[i]["msgs"]), size) if len(chats[i]["msgs"][s:s + size]) >= 4]
    n = len(parts)
    if n < 2:
        return []
    X = np.zeros((n, len(vocab)), np.float32)
    for k, (i, a, b) in enumerate(parts):
        for _, _, t in chats[i]["msgs"][a:b]:
            for w in tokens(t):
                if w in idx: X[k, idx[w]] = 1
    present = X.astype(bool)
    idf = (np.log((1 + n) / (1 + X.sum(0))) + 1).astype(np.float32)
    X *= idf                                                     # X is now the idf-weighted Xw
    Xn = X / np.maximum(X.sum(1, keepdims=True), 1e-9)
    Xs = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    del X
    owner = np.array([p[0] for p in parts])
    root = {}; rid = np.array([root.setdefault(chats[i]["id"].split("#")[0], len(root)) for i in owner])
    Y = (L.astype(np.float32) @ Xn.T)                            # V x n
    nother = np.bincount(rid, minlength=rid.max() + 1)
    nother = (n - nother[rid]).astype(np.float64)
    rowsum = np.zeros(n)
    def rows(lo, hi):
        Mb = Xn[lo:hi] @ Y
        Mb[rid[lo:hi, None] == rid[None, :]] = 0                 # same thread: no link
        return Mb
    for lo in range(0, n, block):                                # pass 1: each part's usual link strength
        rowsum[lo:hi] = rows(lo, hi := min(lo + block, n)).sum(1)
    row = rowsum / max(nother.max(), 1)
    mean = max(row.mean(), 1e-12)
    want = max(top * 400, 20000)
    cand_v, cand_i, cand_j = [], [], []
    for lo in range(0, n, block):                                # pass 2: lift, upper triangle, keep the best
        hi = min(lo + block, n)
        Mb = rows(lo, hi) / np.maximum(np.outer(row[lo:hi], row) / mean, 1e-12)
        Mb[np.arange(hi - lo)[:, None] + lo >= np.arange(n)[None, :]] = 0
        flat = Mb.ravel(); k = min(flat.size - 1, want)
        sel = np.argpartition(-flat, k)[:k]; sel = sel[flat[sel] > 0]
        r, c = np.unravel_index(sel, Mb.shape)
        cand_v.append(flat[sel]); cand_i.append(r + lo); cand_j.append(c)
        v = np.concatenate(cand_v); ii = np.concatenate(cand_i); jj = np.concatenate(cand_j)
        if len(v) > want:
            o = np.argpartition(-v, want)[:want]; v, ii, jj = v[o], ii[o], jj[o]
        cand_v, cand_i, cand_j = [v], [ii], [jj]
    v, ia, ib = cand_v[0], cand_i[0], cand_j[0]
    o = np.lexsort((ib, ia, -v)); v, ia, ib = v[o], ia[o], ib[o]
    best = {}
    for x in range(len(v)):
        a, b = ia[x], ib[x]
        key = (min(rid[a], rid[b]), max(rid[a], rid[b]))         # one link per pair of original threads
        if key not in best:
            best[key] = (a, b, float(v[x]))
            if len(best) >= top: break
    out = []
    for a, b, m in best.values():
        ca, cb = owner[a], owner[b]
        ta, tb = np.flatnonzero(present[a]), np.flatnonzero(present[b]); sub = L[np.ix_(ta, tb)]
        carry = [(vocab[ta[i]], vocab[tb[j]], float(sub[i, j]))
                 for i, j in zip(*np.unravel_index(np.argsort(-sub, axis=None)[:5], sub.shape)) if sub[i, j] > 0]
        A, B = chats[ca], chats[cb]
        w = float(Xs[a] @ Xs[b])
        out.append({"a": A["id"], "a_title": A["title"], "a_acct": A["acct"], "a_msgs": [int(parts[a][1]) + 1, int(parts[a][2])],
                    "b": B["id"], "b_title": B["title"], "b_acct": B["acct"], "b_msgs": [int(parts[b][1]) + 1, int(parts[b][2])],
                    "map": m, "words": w, "hidden": m * (1 - w),
                    "cross_account": A["acct"] != B["acct"], "carried_by": carry})
    return out
