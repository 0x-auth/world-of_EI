"""EI organism v0 — born into a time-ordered stream, learns only from its own prediction errors.

Data: local SQLite copy of D1 `digital-history` (path given by --db; never committed).
See PREREGISTRATION.md for the fixed protocol.
"""
import re, sqlite3, collections, math
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

LIFE = ("2024-06-01", "2025-05-01")
VAL = ("2025-05-01", "2025-06-01")
TEST = ("2025-06-01", "2025-09-01")
SYMBOLS = set("φπΔ∆∞ψΩτθ")
WORD = re.compile(r"[a-z][a-z0-9_'\-]{2,}|[φπΔ∆∞ψΩτθ]")
STOP = set(ENGLISH_STOP_WORDS) | {"don't", "i'm", "it's", "that's", "can't", "you're", "let's", "i've", "i'll"}
V_SIZE, MIN_DF, MAXCH = 4000, 5, 2000


def load_stream(db):
    con = sqlite3.connect(db)
    rows = con.execute("""select sent_at_iso, body_text, 'chat' from messages
        where source_id in (4,5) and role in ('human','user') and sent_at_iso is not null and sent_at_iso != ''
        union all select committed_at, message, 'commit' from github_commits
        where committed_at is not null and committed_at != ''""").fetchall()
    rows = [r for r in rows if r[1]]
    rows.sort(key=lambda r: r[0])
    seen, out = set(), []
    for t, text, kind in rows:
        if len(text) >= 40:
            if text in seen:
                continue
            seen.add(text)
        out.append((t, text[:MAXCH], kind))
    return out


def tokens(text):
    return {w for w in WORD.findall(text.lower()) if w not in STOP}


def build(db):
    stream = [(t, tokens(x), k) for t, x, k in load_stream(db) if t < TEST[1]]
    df = collections.Counter()
    for t, toks, _ in stream:
        if LIFE[0] <= t < LIFE[1]:
            df.update(toks)
    vocab = [w for w, n in df.most_common() if n >= MIN_DF][:V_SIZE]
    idx = {w: i for i, w in enumerate(vocab)}
    msgs = [(t, np.array(sorted(idx[w] for w in toks if w in idx), dtype=np.int32))
            for t, toks, _ in stream if t >= LIFE[0]]
    return vocab, msgs


class Trace:
    """Residue context: c <- a*c + (1-a)*x, L1-normalised, small entries dropped."""
    def __init__(self, V, a):
        self.c = np.zeros(V, np.float32); self.a = a

    def step(self, ids):
        self.c *= self.a
        if len(ids):
            self.c[ids] += (1 - self.a) / len(ids)
        s = self.c.sum()
        if s > 0:
            self.c /= s
        self.c[self.c < 1e-3] = 0
        nz = np.flatnonzero(self.c)
        return nz, self.c[nz]


def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def score(p, ids):
    bits = -np.log2(np.maximum(p[ids], 1e-12)).mean()
    top = np.argpartition(-p, 20)[:20]
    rec = np.isin(ids, top).mean()
    return bits, rec


class Organism:
    def __init__(self, V, eta, seed):
        self.W = np.zeros((V, V), np.float32)
        if seed:
            np.fill_diagonal(self.W, seed)
        self.b = np.zeros(V, np.float32); self.eta = eta

    def predict(self, nz, cv):
        return softmax(self.b + cv @ self.W[nz])

    def learn(self, nz, cv, p, ids):
        y = np.zeros_like(p); y[ids] = 1.0 / len(ids)
        g = (y - p).astype(np.float32) * self.eta
        self.W[nz] += np.outer(cv, g)
        self.b += g


class Counts:
    """Snapshot: unigram + c-weighted next-message co-occurrence counts."""
    def __init__(self, V, lam):
        self.N = np.zeros((V, V), np.float32); self.row = np.zeros(V, np.float32)
        self.df = np.ones(V, np.float64); self.n = V; self.lam = lam

    def predict(self, nz, cv):
        u = self.df / self.n
        if self.lam == 0 or len(nz) == 0:
            return u
        rows = self.N[nz] / np.maximum(self.row[nz], 1)[:, None]
        m = cv @ rows; s = m.sum()
        return self.lam * m / s + (1 - self.lam) * u if s > 0 else u

    def learn(self, prev_ids, ids):
        if len(prev_ids):
            self.N[np.ix_(prev_ids, ids)] += 1; self.row[prev_ids] += len(ids)
        self.df[ids] += 1; self.n += len(ids)


def run(msgs, V, kind, params, eval_window, learn_in_eval):
    """Replay the stream from birth. Learn on life (and val if eval is test). Predict-then-learn ordering."""
    a = params["alpha"]
    model = Organism(V, params["eta"], params["seed"]) if kind == "O" else Counts(V, params.get("lam", 0))
    tr = Trace(V, a)
    prev_ids = np.zeros(0, np.int32); nz = np.zeros(0, np.int64); cv = np.zeros(0, np.float32)
    res = []
    for t, ids in msgs:
        if t >= eval_window[1]:
            break
        in_eval = t >= eval_window[0]
        if len(ids):
            p = model.predict(nz, cv)
            if in_eval:
                res.append((t[:10], *score(p, ids)))
            if (not in_eval) or learn_in_eval:
                if kind == "O":
                    if len(nz):
                        model.learn(nz, cv, p, ids)
                else:
                    model.learn(prev_ids, ids)
            prev_ids = ids
        nz, cv = tr.step(ids)
        nz = nz.copy(); cv = cv.copy()
    return res, model
