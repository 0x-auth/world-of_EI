"""Seed: what one life passes to the next. Corrected weights of both twins + the vocabulary they belong to.
Struck term pairs are cut out of the seed before it is carried, so a struck link is not reborn."""
import numpy as np


def save(path, vocab, fwd, back=None):
    d = {"vocab": np.array(vocab, dtype=object), "Wf": fwd.W.astype(np.float16), "bf": fwd.b}
    if back is not None:
        d.update(Wb=back.W.astype(np.float16), bb=back.b)
    np.savez_compressed(path, **d)


def load(path, vocab, struck=(), base=2.0):
    """Align a carried seed to this run's vocabulary. New words are born as newborns (diagonal = base)."""
    z = np.load(path, allow_pickle=True)
    old = {w: i for i, w in enumerate(z["vocab"].tolist())}
    V = len(vocab)
    src = np.array([old.get(w, -1) for w in vocab]); have = src >= 0
    out = {}
    for key, bkey in (("Wf", "bf"), ("Wb", "bb")):
        if key not in z: out[key] = None; continue
        W = np.zeros((V, V), np.float32); np.fill_diagonal(W, base)
        si = src[have]; di = np.flatnonzero(have)
        W[np.ix_(di, di)] = z[key][np.ix_(si, si)].astype(np.float32)
        b = np.zeros(V, np.float32); b[di] = z[bkey][si]
        idx = {w: i for i, w in enumerate(vocab)}
        for u, v in struck:
            if u in idx and v in idx:
                W[idx[u], idx[v]] = min(W[idx[u], idx[v]], 0); W[idx[v], idx[u]] = min(W[idx[v], idx[u]], 0)
        out[key] = (W, b)
    return out["Wf"], out["Wb"], int(have.sum())
